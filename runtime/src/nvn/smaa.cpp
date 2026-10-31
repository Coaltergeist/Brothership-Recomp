// SMAA 1x on the presented image (smaa.h).
//
// Adapted from drippu's renderer_vulkan/present/smaa.cpp (yuzu's; GPL-2.0-or-later / GPL-3.0-or-later): the same
// three passes (edge detection, blending weights, neighbourhood blending), drippu's compiled shaders and SMAA's
// area / search lookup textures, drippu's present helpers (present/util.h). Recorded into the caller's command
// buffer instead of drippu's scheduler, and the output has the input's format (drippu's: RGBA16F), so the blit
// to the window keeps today's colour encoding (sRGB or not, display.cpp). Made at the first frame and again when
// the image's size or format changes.
#include "smaa.h"

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "vk.h"
#include "../options.h"

#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/smaa_area_tex.h"
#include "video_core/smaa_search_tex.h"
#include "video_core/host_shaders/smaa_blending_weight_calculation_frag_spv.h"
#include "video_core/host_shaders/smaa_blending_weight_calculation_vert_spv.h"
#include "video_core/host_shaders/smaa_edge_detection_frag_spv.h"
#include "video_core/host_shaders/smaa_edge_detection_vert_spv.h"
#include "video_core/host_shaders/smaa_neighborhood_blending_frag_spv.h"
#include "video_core/host_shaders/smaa_neighborhood_blending_vert_spv.h"

namespace rt::nvn {
namespace {

using namespace Vulkan;

enum Stage { kEdges, kWeights, kBlend, kStages };
constexpr u32 kSlots = 4; // frame slots (display.cpp: 2)
constexpr VkFormat kEdgesFormat = VK_FORMAT_R16G16_SFLOAT, kWeightsFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

struct Slot {
    vk::Image edges, weights, output;
    vk::ImageView edges_view, weights_view, output_view, input_view;
    std::array<vk::Framebuffer, kStages> framebuffers;
    vk::DescriptorSets sets;
};

struct Smaa {
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    vk::Image area, search;
    vk::ImageView area_view, search_view;
    vk::Buffer lut_staging; // the lookup textures' bytes (kept: uploaded once, in the first frame's commands)
    vk::Sampler sampler;
    std::array<vk::ShaderModule, kStages> vert, frag;
    std::array<vk::RenderPass, kStages> passes;
    std::array<vk::DescriptorSetLayout, kStages> set_layouts;
    std::array<vk::PipelineLayout, kStages> layouts;
    std::array<vk::Pipeline, kStages> pipelines;
    vk::DescriptorPool pool;
    std::array<Slot, kSlots> slots;
    bool luts_uploaded = false;
};

std::mutex g_mu;
std::unique_ptr<Smaa> g_smaa;

/// The output image: the input's format, usable as a render target, sampled, and the blit's source.
vk::Image OutputImage(VkExtent2D extent, VkFormat format) {
    return Vk().allocator->CreateImage({
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
}

std::unique_ptr<Smaa> Make(VkExtent2D extent, VkFormat format) {
    const Device& device = *Vk().device;
    MemoryAllocator& allocator = *Vk().allocator;
    auto s = std::make_unique<Smaa>();
    s->extent = extent, s->format = format;
    s->area = CreateWrappedImage(allocator, {AREATEX_WIDTH, AREATEX_HEIGHT}, VK_FORMAT_R8G8_UNORM);
    s->search = CreateWrappedImage(allocator, {SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT}, VK_FORMAT_R8_UNORM);
    s->area_view = CreateWrappedImageView(device, s->area, VK_FORMAT_R8G8_UNORM);
    s->search_view = CreateWrappedImageView(device, s->search, VK_FORMAT_R8_UNORM);
    s->sampler = CreateWrappedSampler(device);
    const std::array<std::span<const u32>, kStages> vert{ARRAY_TO_SPAN(SMAA_EDGE_DETECTION_VERT_SPV),
                                                          ARRAY_TO_SPAN(SMAA_BLENDING_WEIGHT_CALCULATION_VERT_SPV),
                                                          ARRAY_TO_SPAN(SMAA_NEIGHBORHOOD_BLENDING_VERT_SPV)};
    const std::array<std::span<const u32>, kStages> frag{ARRAY_TO_SPAN(SMAA_EDGE_DETECTION_FRAG_SPV),
                                                          ARRAY_TO_SPAN(SMAA_BLENDING_WEIGHT_CALCULATION_FRAG_SPV),
                                                          ARRAY_TO_SPAN(SMAA_NEIGHBORHOOD_BLENDING_FRAG_SPV)};
    const std::array<VkFormat, kStages> formats{kEdgesFormat, kWeightsFormat, format};
    // descriptors: edges 1 (input), weights 3 (edges, area, search), blend 2 (input, weights)
    constexpr VkDescriptorType kSampled = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    s->set_layouts[kEdges] = CreateWrappedDescriptorSetLayout(device, {kSampled});
    s->set_layouts[kWeights] = CreateWrappedDescriptorSetLayout(device, {kSampled, kSampled, kSampled});
    s->set_layouts[kBlend] = CreateWrappedDescriptorSetLayout(device, {kSampled, kSampled});
    for (u32 i = 0; i < kStages; ++i) {
        s->vert[i] = CreateWrappedShaderModule(device, vert[i]);
        s->frag[i] = CreateWrappedShaderModule(device, frag[i]);
        s->passes[i] = CreateWrappedRenderPass(device, formats[i]);
        s->layouts[i] = CreateWrappedPipelineLayout(device, s->set_layouts[i]);
        s->pipelines[i] = CreateWrappedPipeline(device, s->passes[i], s->layouts[i], std::tie(s->vert[i], s->frag[i]));
    }
    s->pool = CreateWrappedDescriptorPool(device, 6 * kSlots, kStages * kSlots);
    std::vector<VkDescriptorSetLayout> set_layouts;
    for (const auto& l : s->set_layouts) set_layouts.push_back(*l);
    for (Slot& slot : s->slots) {
        slot.edges = CreateWrappedImage(allocator, extent, kEdgesFormat);
        slot.weights = CreateWrappedImage(allocator, extent, kWeightsFormat);
        slot.output = OutputImage(extent, format);
        slot.edges_view = CreateWrappedImageView(device, slot.edges, kEdgesFormat);
        slot.weights_view = CreateWrappedImageView(device, slot.weights, kWeightsFormat);
        slot.output_view = CreateWrappedImageView(device, slot.output, format);
        slot.framebuffers[kEdges] = CreateWrappedFramebuffer(device, s->passes[kEdges], slot.edges_view, extent);
        slot.framebuffers[kWeights] = CreateWrappedFramebuffer(device, s->passes[kWeights], slot.weights_view, extent);
        slot.framebuffers[kBlend] = CreateWrappedFramebuffer(device, s->passes[kBlend], slot.output_view, extent);
        slot.sets = CreateWrappedDescriptorSets(s->pool, set_layouts);
    }
    Log("nvn: SMAA at %ux%u (format %d; RT_AA)", extent.width, extent.height, format);
    return s;
}

/// The lookup textures' upload and the slots' images cleared (SMAA's edge pass writes edge pixels only), in the
/// first frame's commands.
void Prepare(Smaa& s, vk::CommandBuffer& cmd) {
    if (s.luts_uploaded) return;
    const size_t area_size = sizeof areaTexBytes, search_size = sizeof searchTexBytes;
    s.lut_staging = CreateWrappedBuffer(*Vk().allocator, area_size + search_size, MemoryUsage::Upload);
    std::memcpy(s.lut_staging.Mapped().data(), areaTexBytes, area_size);
    std::memcpy(s.lut_staging.Mapped().data() + area_size, searchTexBytes, search_size);
    s.lut_staging.Flush();
    const auto upload = [&](VkImage image, VkDeviceSize offset, VkExtent2D e) {
        TransitionImageLayout(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED);
        cmd.CopyBufferToImage(*s.lut_staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VkBufferImageCopy{offset, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                                                {e.width, e.height, 1}});
        TransitionImageLayout(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    };
    upload(*s.area, 0, {AREATEX_WIDTH, AREATEX_HEIGHT});
    upload(*s.search, area_size, {SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT});
    for (Slot& slot : s.slots) {
        for (VkImage image : {*slot.edges, *slot.weights, *slot.output}) ClearColorImage(cmd, image);
    }
    FullBarrier(cmd);
    s.luts_uploaded = true;
}

} // namespace

bool SmaaOn() {
    static const bool on = [] {
        const char* v = options::Get("RT_AA");
        if (!v || std::strcmp(v, "off") == 0) return false;
        if (std::strcmp(v, "smaa") != 0) Fatal("RT_AA=%s: expected off or smaa", v);
        return true;
    }();
    return on;
}

VkImage SmaaApply(vk::CommandBuffer cmd, u32 slot_index, const TextureImage& input, VkExtent2D extent) {
    std::lock_guard lock(g_mu);
    if (slot_index >= kSlots) Fatal("nvn: SMAA: frame slot %u of %u", slot_index, kSlots);
    const Device& device = *Vk().device;
    if (!g_smaa || g_smaa->extent.width != extent.width || g_smaa->extent.height != extent.height ||
        g_smaa->format != input.format) {
        if (g_smaa) device.GetLogical().WaitIdle(); // (a new size or format: rare)
        g_smaa = Make(extent, input.format);
    }
    Smaa& s = *g_smaa;
    Prepare(s, cmd);
    Slot& slot = s.slots[slot_index];
    slot.input_view = CreateWrappedImageView(device, const_cast<vk::Image&>(input.image), input.format);
    std::vector<VkDescriptorImageInfo> infos;
    infos.reserve(6);
    std::vector<VkWriteDescriptorSet> writes;
    writes.push_back(CreateWriteDescriptorSet(infos, *s.sampler, *slot.input_view, slot.sets[kEdges], 0));
    writes.push_back(CreateWriteDescriptorSet(infos, *s.sampler, *slot.edges_view, slot.sets[kWeights], 0));
    writes.push_back(CreateWriteDescriptorSet(infos, *s.sampler, *s.area_view, slot.sets[kWeights], 1));
    writes.push_back(CreateWriteDescriptorSet(infos, *s.sampler, *s.search_view, slot.sets[kWeights], 2));
    writes.push_back(CreateWriteDescriptorSet(infos, *s.sampler, *slot.input_view, slot.sets[kBlend], 0));
    writes.push_back(CreateWriteDescriptorSet(infos, *s.sampler, *slot.weights_view, slot.sets[kBlend], 1));
    device.GetLogical().UpdateDescriptorSets(writes, {});
    for (u32 i = 0; i < kStages; ++i) {
        BeginRenderPass(cmd, *s.passes[i], *slot.framebuffers[i], extent);
        cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *s.pipelines[i]);
        cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *s.layouts[i], 0, slot.sets[i], {});
        cmd.Draw(3, 1, 0, 0);
        cmd.EndRenderPass();
        FullBarrier(cmd);
    }
    return *slot.output;
}

} // namespace rt::nvn
