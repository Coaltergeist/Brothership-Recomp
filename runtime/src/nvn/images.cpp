// Texture images, the upload ring and copies into textures.
//
// Texture contents live in Vulkan images (6.3), one per texture storage
// address, created at first use from the texture's settings; storage
// initialised again with other settings gets a new image (the old one lives
// until the submissions using it retire). Images stay in
// VK_IMAGE_LAYOUT_GENERAL. Not yet: contents the CPU wrote into a texture's
// storage (6.3's deswizzle and write tracking), multisample, cube arrays,
// copies into packed depth + stencil formats, views with another format.
//
// nvnCommandBufferCopyBufferToTexture (BRO 0x565550, read 2026-10-16): the
// region is in texels and rounded up to format blocks (x, y, width, height);
// the buffer rows are tightly packed unless SetCopyRowStride /
// SetCopyImageStride were set (row: width in blocks x bytes per block; image:
// row x height in blocks) — Brothership never sets them, and the layer stops
// if a game does (not implemented); for 1D arrays y / height select layers,
// otherwise z / depth (3D slices, array layers, cube faces). The source rows
// are copied at submit (5.0 rule 1).
#include "images.h"

#include "../resolution.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace rt::nvn {

// --- Formats (6.1) -------------------------------------------------------------------------------
// After LASRecomp's textures.c (github.com/peiikai/LASRecomp, 6244eb9, MIT),
// keeping only exact matches: its stand-ins (24-bit depth as 32-bit float,
// RGB8/16/32 expanded on upload, the 16-bit packed colour formats) have no
// format here until a game needs them. X formats (alpha ignored) map to the
// format with alpha; the view's swizzle handles them when textures are sampled.

VkFormat VulkanFormat(u32 id) {
    switch (id) {
    case 0x01: return VK_FORMAT_R8_UNORM;
    case 0x02: return VK_FORMAT_R8_SNORM;
    case 0x03: return VK_FORMAT_R8_UINT;
    case 0x04: return VK_FORMAT_R8_SINT;
    case 0x05: return VK_FORMAT_R16_SFLOAT;
    case 0x06: return VK_FORMAT_R16_UNORM;
    case 0x07: return VK_FORMAT_R16_SNORM;
    case 0x08: return VK_FORMAT_R16_UINT;
    case 0x09: return VK_FORMAT_R16_SINT;
    case 0x0a: return VK_FORMAT_R32_SFLOAT;
    case 0x0b: return VK_FORMAT_R32_UINT;
    case 0x0c: return VK_FORMAT_R32_SINT;
    case 0x0d: return VK_FORMAT_R8G8_UNORM;
    case 0x0e: return VK_FORMAT_R8G8_SNORM;
    case 0x0f: return VK_FORMAT_R8G8_UINT;
    case 0x10: return VK_FORMAT_R8G8_SINT;
    case 0x11: return VK_FORMAT_R16G16_SFLOAT;
    case 0x12: return VK_FORMAT_R16G16_UNORM;
    case 0x13: return VK_FORMAT_R16G16_SNORM;
    case 0x14: return VK_FORMAT_R16G16_UINT;
    case 0x15: return VK_FORMAT_R16G16_SINT;
    case 0x16: return VK_FORMAT_R32G32_SFLOAT;
    case 0x17: return VK_FORMAT_R32G32_UINT;
    case 0x18: return VK_FORMAT_R32G32_SINT;
    case 0x25: return VK_FORMAT_R8G8B8A8_UNORM;
    case 0x26: return VK_FORMAT_R8G8B8A8_SNORM;
    case 0x27: return VK_FORMAT_R8G8B8A8_UINT;
    case 0x28: return VK_FORMAT_R8G8B8A8_SINT;
    case 0x29: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case 0x2a: return VK_FORMAT_R16G16B16A16_UNORM;
    case 0x2b: return VK_FORMAT_R16G16B16A16_SNORM;
    case 0x2c: return VK_FORMAT_R16G16B16A16_UINT;
    case 0x2d: return VK_FORMAT_R16G16B16A16_SINT;
    case 0x2e: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case 0x2f: return VK_FORMAT_R32G32B32A32_UINT;
    case 0x30: return VK_FORMAT_R32G32B32A32_SINT;
    case 0x31: return VK_FORMAT_S8_UINT;
    case 0x32: return VK_FORMAT_D16_UNORM;
    case 0x34: return VK_FORMAT_D32_SFLOAT;
    case 0x35: return VK_FORMAT_D24_UNORM_S8_UINT;
    case 0x36: return VK_FORMAT_D32_SFLOAT_S8_UINT;
    case 0x37: // RGBX8 sRGB
    case 0x38: return VK_FORMAT_R8G8B8A8_SRGB;
    case 0x3d: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 0x3e: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case 0x3f: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case 0x40: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case 0x41: return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    case 0x42: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case 0x43: return VK_FORMAT_BC2_UNORM_BLOCK;
    case 0x44: return VK_FORMAT_BC3_UNORM_BLOCK;
    case 0x45: return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
    case 0x46: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case 0x47: return VK_FORMAT_BC2_SRGB_BLOCK;
    case 0x48: return VK_FORMAT_BC3_SRGB_BLOCK;
    case 0x49: return VK_FORMAT_BC4_UNORM_BLOCK;
    case 0x4a: return VK_FORMAT_BC4_SNORM_BLOCK;
    case 0x4b: return VK_FORMAT_BC5_UNORM_BLOCK;
    case 0x4c: return VK_FORMAT_BC5_SNORM_BLOCK;
    case 0x4d: return VK_FORMAT_BC7_UNORM_BLOCK;
    case 0x4e: return VK_FORMAT_BC7_SRGB_BLOCK;
    case 0x4f: return VK_FORMAT_BC6H_SFLOAT_BLOCK;
    case 0x50: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
    case 0x99: // BGRX8 (ids 0x99-0x9c: formats.cpp)
    case 0x9a: return VK_FORMAT_B8G8R8A8_UNORM;
    case 0x9b: // BGRX8 sRGB
    case 0x9c: return VK_FORMAT_B8G8R8A8_SRGB;
    default:
        if (id >= 0x79 && id <= 0x86) return static_cast<VkFormat>(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * (id - 0x79));
        if (id >= 0x87 && id <= 0x94) return static_cast<VkFormat>(VK_FORMAT_ASTC_4x4_SRGB_BLOCK + 2 * (id - 0x87));
        return VK_FORMAT_UNDEFINED;
    }
}

// --- Images (6.3) ---------------------------------------------------------------------------------

namespace {

constexpr u32 kTarget1D = 0, kTarget2D = 1, kTarget3D = 2, kTarget1DArray = 3, kTarget2DArray = 4,
              kTargetRectangle = 7, kTargetCube = 8;

std::mutex g_images_mu;
std::map<u64, std::shared_ptr<TextureImage>> g_images; // by storage address

bool SameSettings(const TextureBuilder& a, const TextureBuilder& b) {
    return a.target == b.target && a.width == b.width && a.height == b.height && a.depth == b.depth &&
           a.levels == b.levels && a.format == b.format && a.flags == b.flags && a.samples == b.samples;
}

/// whether an image of these settings is drawn at the run's scale. As yuzu's rule
/// (drippu's texture_cache/image_info.cpp): a 2D image of one level and one sample, taller than 288 texels
/// (RT_RES_MIN_HEIGHT), that can be a render target and is not linear (the CPU's layout). Decided once, when the
/// image is made (yuzu scales images up and down as they are used): the plaza's screen-sized targets, its blur
/// chains down to 640x360 / 320x360, the menus' 1280x1024 scene and the battle's 1920x1080 target qualify; cube
/// maps, 3D tables, mipmapped textures and compressed formats never do.
bool ScalesWith(const TextureBuilder& b, VkImageUsageFlags usage) {
    static const u32 min_height = [] {
        const char* e = std::getenv("RT_RES_MIN_HEIGHT");
        return e && *e ? static_cast<u32>(std::atoi(e)) : 288u;
    }();
    if (ScaleN() == 4 || b.target != kTarget2D || std::max<u32>(b.levels, 1) != 1 || b.samples > 1) return false;
    if (!(usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) return false;
    if ((b.flags & 0x10) || b.height <= min_height) return false; // linear; small
    const u32 max = Vk().device->GetPhysical().GetProperties().limits.maxImageDimension2D;
    return u64{b.width} * ScaleN() / 4 <= max && u64{b.height} * ScaleN() / 4 <= max;
}

/// Logs each kind of scaled image once.
void NoteScaled(const TextureBuilder& b, const TextureImage& img) {
    static std::mutex mu;
    static std::set<std::array<u32, 3>> seen;
    std::lock_guard lock(mu);
    if (seen.insert({b.width, b.height, b.format}).second) {
        Log("resolution: %ux%u images of format 0x%x are drawn at %ux%u", b.width, b.height, b.format,
            img.UpSize(b.width), img.UpSize(b.height));
    }
}

std::shared_ptr<TextureImage> MakeImage(const TextureState& t, bool may_scale = true) {
    const TextureBuilder& b = t.settings;
    if (b.samples > 1) Fatal("nvn: multisample texture images are not implemented yet");
    VkFormat format = VulkanFormat(b.format);
    if (format == VK_FORMAT_UNDEFINED) Fatal("nvn: no Vulkan format for NVN format 0x%x yet", b.format);
    constexpr VkFormatFeatureFlags kNeeded = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                             VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    const auto features_of = [](VkFormat f) {
        return Vk().device->GetPhysical().GetFormatProperties(f).optimalTilingFeatures;
    };
    // AMD has no 24-bit depth (RADV on the Steam Deck, 2026-11-02): D32F + S8 instead, as yuzu's
    // vulkan_device.cpp alternatives. Nothing reads the image's bytes as D24S8 (copies and views of
    // another format stop the run above), so only depth precision differs. RT_NVN_D32S8=1: always (to compare
    // on a device that has D24S8).
    static const bool force_d32s8 = std::getenv("RT_NVN_D32S8") && *std::getenv("RT_NVN_D32S8") == '1';
    if (format == VK_FORMAT_D24_UNORM_S8_UINT && (force_d32s8 || (features_of(format) & kNeeded) != kNeeded)) {
        static std::once_flag told;
        std::call_once(told, [] { Log("nvn: depth 24 + stencil 8 images are made as D32F + S8 (the device has no D24S8)"); });
        format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    }
    const VkFormatFeatureFlags features = features_of(format);
    if ((features & kNeeded) != kNeeded) {
        Fatal("nvn: the Vulkan device cannot sample or copy format %d (NVN 0x%x)", format, b.format);
    }
    VkImageUsageFlags usage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (features & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) usage |= VK_IMAGE_USAGE_STORAGE_BIT;

    VkImageType type = VK_IMAGE_TYPE_2D;
    VkExtent3D extent{b.width, b.height, 1};
    u32 layers = 1;
    VkImageCreateFlags flags = 0;
    switch (b.target) {
    case kTarget1D: type = VK_IMAGE_TYPE_1D, extent = {b.width, 1, 1}; break;
    case kTarget1DArray: type = VK_IMAGE_TYPE_1D, extent = {b.width, 1, 1}, layers = b.height; break;
    case kTarget2D:
    case kTargetRectangle: break;
    case kTarget2DArray: layers = b.depth; break;
    case kTargetCube: layers = 6, flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; break;
    case kTarget3D: // render targets: slices attached as a 2D array (render.cpp)
        type = VK_IMAGE_TYPE_3D, extent.depth = b.depth;
        if (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
        }
        break;
    default: Fatal("nvn: images of texture target %u are not implemented yet", b.target);
    }
    auto img = std::make_shared<TextureImage>();
    img->state = t;
    img->storage = (usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
    if (may_scale && ScalesWith(b, usage)) {
        img->scale_n = ScaleN();
        extent.width = img->UpSize(extent.width), extent.height = img->UpSize(extent.height);
        NoteScaled(b, *img);
    }
    img->format = format;
    switch (format) {
    case VK_FORMAT_S8_UINT: img->aspect = VK_IMAGE_ASPECT_STENCIL_BIT; break;
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT: img->aspect = VK_IMAGE_ASPECT_DEPTH_BIT; break;
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: img->aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT; break;
    default: img->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    img->image = Vk().allocator->CreateImage({
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .imageType = type,
        .format = format,
        .extent = extent,
        .mipLevels = std::max<u32>(b.levels, 1),
        .arrayLayers = std::max<u32>(layers, 1),
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    return img;
}

} // namespace

namespace {
std::mutex g_sampled_mu; // TextureImage::sampled of every image

/// Formats whose fourth channel is padding ("X"): sampled, it reads as 1.
bool AlphaIsOne(u32 format) { return format == 0x37 || format == 0x6d || format == 0x99 || format == 0x9b; }

VkComponentSwizzle Swizzle(u8 nvn) {
    if (nvn > 5) Fatal("nvn: texture swizzle %u", nvn);
    return static_cast<VkComponentSwizzle>(nvn + 1); // NVN 0 zero, 1 one, 2-5 r g b a: Vulkan's value - 1
}
} // namespace

VkImageView SampledView(TextureImage& img, const TextureState& t, const ViewArg& view, VkImageViewType type) {
    const TextureBuilder& b = t.settings;
    const TextureView& v = view.view;
    const bool has = view.present != 0;
    if (has && (v.set & kViewFormat) && v.format != b.format) {
        Fatal("nvn: sampling texture format 0x%x through a view of format 0x%x is not implemented yet", b.format, v.format);
    }
    const u32 levels = std::max<u32>(b.levels, 1);
    u32 level0 = has && (v.set & kViewLevels) ? v.base_level : 0, nlevels = levels - std::min(level0, levels - 1);
    if (has && (v.set & kViewLevels) && v.num_levels > 0) nlevels = v.num_levels;
    if (level0 >= levels || level0 + nlevels > levels) Fatal("nvn: view levels %u+%u of %u", level0, nlevels, levels);
    const u32 all_layers = b.target == kTarget2DArray ? b.depth : b.target == kTargetCube ? 6 : b.target == kTarget1DArray ? b.height : 1;
    u32 layer0 = has && (v.set & kViewLayers) ? static_cast<u32>(v.base_layer) : 0;
    u32 nlayers = all_layers - std::min(layer0, all_layers);
    if (has && (v.set & kViewLayers) && v.num_layers > 0) nlayers = static_cast<u32>(v.num_layers);
    if (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_3D) nlayers = 1;
    if (type == VK_IMAGE_VIEW_TYPE_CUBE) nlayers = 6;
    if (nlayers == 0 || layer0 + nlayers > all_layers) Fatal("nvn: view layers %u+%u of %u", layer0, nlayers, all_layers);
    VkImageAspectFlags aspect = img.aspect;
    if (aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
        const u32 ds = has && (v.set & kViewDsMode) ? (v.set >> 6) & 3 : b.ds_mode; // 0 depth, 1 stencil
        aspect = ds == 1 ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    const u8* sw = has && (v.set & kViewSwizzle) ? v.swizzle : b.swizzle;
    VkComponentMapping comp{};
    if (img.aspect == VK_IMAGE_ASPECT_COLOR_BIT && (sw[0] | sw[1] | sw[2] | sw[3])) { // all zero: never set
        comp = {Swizzle(sw[0]), Swizzle(sw[1]), Swizzle(sw[2]), Swizzle(sw[3])};
        if (AlphaIsOne(b.format)) {
            for (VkComponentSwizzle* c : {&comp.r, &comp.g, &comp.b, &comp.a}) {
                if (*c == VK_COMPONENT_SWIZZLE_A) *c = VK_COMPONENT_SWIZZLE_ONE;
            }
        }
    } else if (AlphaIsOne(b.format)) {
        comp.a = VK_COMPONENT_SWIZZLE_ONE;
    }
    const std::array<u32, 7> key{static_cast<u32>(type), level0 | nlevels << 16, layer0 | nlayers << 16,
                                 static_cast<u32>(aspect), static_cast<u32>(comp.r) | static_cast<u32>(comp.g) << 8,
                                 static_cast<u32>(comp.b) | static_cast<u32>(comp.a) << 8, 0};
    std::lock_guard lock(g_sampled_mu);
    auto& out = img.sampled[key];
    if (!out) {
        out = Vk().device->GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *img.image,
            .viewType = type,
            .format = img.format,
            .components = comp,
            .subresourceRange = {aspect, level0, nlevels, layer0, nlayers},
        });
    }
    return *out;
}

VkImageView StorageView(TextureImage& img, const TextureState& t, const ViewArg& view, VkImageViewType type) {
    const TextureBuilder& b = t.settings;
    const TextureView& v = view.view;
    const bool has = view.present != 0;
    if (!img.storage) Fatal("nvn: storage image of format 0x%x, which the Vulkan device cannot store to", b.format);
    if (has && (v.set & kViewFormat) && v.format != b.format) {
        Fatal("nvn: an image of format 0x%x through a view of format 0x%x is not implemented yet", b.format, v.format);
    }
    const u32 level = has && (v.set & kViewLevels) ? v.base_level : 0;
    const u32 all_layers = b.target == kTarget2DArray ? b.depth : b.target == kTargetCube ? 6 : b.target == kTarget1DArray ? b.height : 1;
    u32 layer0 = has && (v.set & kViewLayers) ? static_cast<u32>(v.base_layer) : 0;
    u32 nlayers = all_layers - std::min(layer0, all_layers);
    if (has && (v.set & kViewLayers) && v.num_layers > 0) nlayers = static_cast<u32>(v.num_layers);
    if (type != VK_IMAGE_VIEW_TYPE_1D_ARRAY && type != VK_IMAGE_VIEW_TYPE_2D_ARRAY && type != VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) nlayers = 1;
    if (type == VK_IMAGE_VIEW_TYPE_CUBE) nlayers = 6;
    const std::array<u32, 7> key{static_cast<u32>(type), level | 1u << 16, layer0 | nlayers << 16, 0, 0, 0, 1};
    std::lock_guard lock(g_sampled_mu);
    auto& out = img.sampled[key];
    if (!out) {
        out = Vk().device->GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *img.image,
            .viewType = type,
            .format = img.format,
            .components = {},
            .subresourceRange = {img.aspect, level, 1, layer0, nlayers},
        });
    }
    return *out;
}

u32 ScaleN() {
    static const u32 n = [] {
        const resolution::Scale& s = resolution::Get();
        return s.up_scale * 4 >> s.down_shift;
    }();
    return n;
}

namespace {
/// The image's layout transition, recorded into `cmd` at its first use.
void Initialise(TextureImage& img, const Vulkan::vk::CommandBuffer& cmd) {
    if (img.initialised) return;
    EndOpenPass(cmd);
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                        VkImageMemoryBarrier{
                            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            .pNext = nullptr,
                            .srcAccessMask = 0,
                            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = *img.image,
                            .subresourceRange = {img.aspect, 0, VK_REMAINING_MIP_LEVELS, 0,
                                                 VK_REMAINING_ARRAY_LAYERS},
                        });
    img.initialised = true;
}

/// An unscaled image of `t`'s settings, initialised: copies between the game's size and a scaled image go
/// through it.
std::shared_ptr<TextureImage> Twin(const TextureState& t, const Vulkan::vk::CommandBuffer& cmd) {
    auto img = MakeImage(t, false);
    Initialise(*img, cmd);
    return img;
}

/// A region in the game's units as `img`'s blit corners (its own units).
std::array<VkOffset3D, 2> Corners(const TextureImage& img, VkOffset3D o, VkExtent3D e) {
    return {VkOffset3D{img.Up(o.x), img.Up(o.y), o.z},
            VkOffset3D{img.Up(o.x + static_cast<s32>(e.width)), img.Up(o.y + static_cast<s32>(e.height)),
                       o.z + static_cast<s32>(e.depth)}};
}
/// The same region (the game's units) blitted from `s` to `d`, each at its own scale.
void BlitArea(const Vulkan::vk::CommandBuffer& cmd, const TextureImage& s, const TextureImage& d,
              const VkImageSubresourceLayers& sub, VkOffset3D o, VkExtent3D e, VkFilter filter) {
    const auto sc = Corners(s, o, e), dc = Corners(d, o, e);
    const VkImageBlit blit{sub, {sc[0], sc[1]}, sub, {dc[0], dc[1]}};
    Vk().device->GetDispatchLoader().vkCmdBlitImage(*cmd, *s.image, VK_IMAGE_LAYOUT_GENERAL, *d.image,
                                                    VK_IMAGE_LAYOUT_GENERAL, 1, &blit, filter);
}
} // namespace

std::shared_ptr<TextureImage> ImageFor(const TextureState& t, const Vulkan::vk::CommandBuffer& cmd) {
    std::lock_guard lock(g_images_mu);
    auto& slot = g_images[t.address];
    if (!slot || !SameSettings(slot->state.settings, t.settings)) slot = MakeImage(t);
    Initialise(*slot, cmd);
    return slot;
}

namespace {

u32 CeilDiv(u32 a, u32 b) { return (a + b - 1) / b; }

} // namespace

namespace {
struct OpenPass {
    VkCommandBuffer cmd;
    VkRenderPass pass;
    VkFramebuffer framebuffer;
    VkExtent2D extent;
};
std::mutex g_open_mu;
std::vector<OpenPass> g_open; // one per command buffer being recorded (a queue's translation)
std::vector<std::pair<VkCommandBuffer, SampleCounting*>> g_counting; // with g_open_mu

SampleCounting* CountingOf(const Vulkan::vk::CommandBuffer& cmd) {
    std::lock_guard lock(g_open_mu);
    for (const auto& [c, counting] : g_counting) {
        if (c == *cmd) return counting;
    }
    return nullptr;
}
} // namespace

void SetSampleCounting(const Vulkan::vk::CommandBuffer& cmd, SampleCounting* counting) {
    std::lock_guard lock(g_open_mu);
    std::erase_if(g_counting, [&](const auto& e) { return e.first == *cmd; });
    if (counting) g_counting.push_back({*cmd, counting});
}

std::atomic<u64> g_pass_draws{0}, g_passes_begun{0};

/// Logs how often draws share a pass, every 10,000 draws.
void CountDrawPass(bool kept) {
    if (!kept) g_passes_begun.fetch_add(1, std::memory_order_relaxed);
    const u64 n = g_pass_draws.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n % 10000 == 0) {
        Log("nvn draw: %llu draws in %llu render passes", (unsigned long long)n,
            (unsigned long long)g_passes_begun.load(std::memory_order_relaxed));
    }
}

void BeginDrawPass(const Vulkan::vk::CommandBuffer& cmd, const VkRenderPassBeginInfo& begin) {
    {
        std::lock_guard lock(g_open_mu);
        for (const OpenPass& o : g_open) {
            if (o.cmd != *cmd) continue;
            if (o.pass == begin.renderPass && o.framebuffer == begin.framebuffer &&
                o.extent.width == begin.renderArea.extent.width && o.extent.height == begin.renderArea.extent.height) {
                CountDrawPass(true);
                return;
            }
            break;
        }
    }
    CountDrawPass(false);
    EndOpenPass(cmd);
    SampleCounting* counting = CountingOf(cmd);
    if (counting) counting->Pause();
    cmd.BeginRenderPass(begin, VK_SUBPASS_CONTENTS_INLINE);
    if (counting) counting->Resume();
    std::lock_guard lock(g_open_mu);
    g_open.push_back({*cmd, begin.renderPass, begin.framebuffer, begin.renderArea.extent});
}

bool PassOpen(const Vulkan::vk::CommandBuffer& cmd) {
    std::lock_guard lock(g_open_mu);
    return std::any_of(g_open.begin(), g_open.end(), [&](const OpenPass& o) { return o.cmd == *cmd; });
}

bool EndOpenPass(const Vulkan::vk::CommandBuffer& cmd) {
    {
        std::lock_guard lock(g_open_mu);
        const auto it = std::find_if(g_open.begin(), g_open.end(), [&](const OpenPass& o) { return o.cmd == *cmd; });
        if (it == g_open.end()) return false;
        g_open.erase(it);
    }
    SampleCounting* counting = CountingOf(cmd);
    if (counting) counting->Pause();
    cmd.EndRenderPass();
    if (counting) counting->Resume();
    // what the pass's draws wrote, before what comes next (as the per-draw barrier did)
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                        VkMemoryBarrier{
                            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .pNext = nullptr,
                            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                        });
    return true;
}

void FullBarrier(const Vulkan::vk::CommandBuffer& cmd) {
    if (EndOpenPass(cmd)) return; // that was the same barrier
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                        VkMemoryBarrier{
                            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .pNext = nullptr,
                            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                        });
}

std::unique_ptr<ImageDownload> StartDownload(const TextureState& t, u32 level) {
    std::shared_ptr<TextureImage> img;
    {
        std::lock_guard lock(g_images_mu);
        auto it = g_images.find(t.address);
        if (it == g_images.end() || !SameSettings(it->second->state.settings, t.settings)) return nullptr;
        img = it->second;
    }
    const TextureBuilder& b = t.settings;
    const FormatInfo* f = FindFormat(b.format);
    const bool sliced = b.target == kTarget3D, array1d = b.target == kTarget1DArray;
    u32 lw = std::max(1u, b.width >> level), lh = array1d ? 1 : std::max(1u, b.height >> level);
    if (img->Scaled()) lw = img->UpSize(lw), lh = img->UpSize(lh); // (one level, 2D)
    const u32 count = sliced ? std::max(1u, b.depth >> level)
                             : b.target == kTarget2DArray ? b.depth : b.target == kTargetCube ? 6 : array1d ? b.height : 1;
    const u32 wb = CeilDiv(lw, f->block_w), hb = CeilDiv(lh, f->block_h);
    const u64 size = u64{wb} * hb * f->bytes * count;
    const auto& dev = *Vk().device;
    auto d = std::make_unique<ImageDownload>();
    d->size = size;
    d->width = lw, d->height = lh;
    d->image = img;
    d->buffer = Vk().allocator->CreateBuffer(
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        Vulkan::MemoryUsage::Download);
    d->pool = dev.GetLogical().CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = dev.GetGraphicsFamily(),
    });
    d->buffers = d->pool.Allocate(1);
    d->done = dev.GetLogical().CreateFence({.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = nullptr, .flags = 0});
    const Vulkan::vk::CommandBuffer cmd(d->buffers[0], dev.GetDispatchLoader());
    cmd.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    FullBarrier(cmd);
    cmd.CopyImageToBuffer(*img->image, VK_IMAGE_LAYOUT_GENERAL, *d->buffer,
                          VkBufferImageCopy{
                              .bufferOffset = 0,
                              .bufferRowLength = wb * f->block_w,
                              .bufferImageHeight = hb * f->block_h,
                              .imageSubresource = {img->aspect, level, 0, sliced ? 1 : count},
                              .imageOffset = {0, 0, 0},
                              .imageExtent = {lw, lh, sliced ? count : 1},
                          });
    FullBarrier(cmd);
    cmd.End();
    const VkSubmitInfo info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = cmd.address(),
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    {
        std::lock_guard queue_lock(QueueLock());
        if (dev.GetGraphicsQueue().Submit(info, *d->done) != VK_SUCCESS) Fatal("nvn: StartDownload: vkQueueSubmit failed");
    }
    return d;
}

std::vector<u8> ImageDownload::Take() {
    if (done.Wait() != VK_SUCCESS) Fatal("nvn: ImageDownload: vkWaitForFences failed");
    buffer.Invalidate();
    return {buffer.Mapped().begin(), buffer.Mapped().begin() + size};
}

std::vector<u8> DownloadImage(const TextureState& t, u32 level) {
    const auto d = StartDownload(t, level);
    return d ? d->Take() : std::vector<u8>{};
}

// --- Upload ring ----------------------------------------------------------------------------------

UploadRing::Slice UploadRing::Allocate(u64 size, u64 alignment, u64 min_offset) {
    constexpr u64 kChunk = 4ULL << 20;
    min_offset = (min_offset + alignment - 1) / alignment * alignment;
    for (;;) {
        if (current_ < chunks_.size()) {
            const u64 at = std::max((used_ + alignment - 1) / alignment * alignment, min_offset);
            if (at + size <= sizes_[current_]) {
                used_ = at + size;
                const auto mapped = chunks_[current_].Mapped(); // empty for device memory the CPU cannot map
                return {*chunks_[current_], at, mapped.empty() ? nullptr : mapped.data() + at};
            }
            if (current_ + 1 < chunks_.size()) {
                ++current_, used_ = 0;
                continue;
            }
        }
        sizes_.push_back(std::max(kChunk, min_offset + size));
        chunks_.push_back(Vk().allocator->CreateBuffer(
            {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = sizes_.back(),
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            usage_));
        current_ = chunks_.size() - 1, used_ = 0;
    }
}

VkBufferView UploadRing::TexelView(const Slice& slice, u64 range, VkFormat format) {
    auto& view = views_[{slice.buffer, slice.offset, range, format}];
    if (!view) {
        view = Vk().device->GetLogical().CreateBufferView({
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .buffer = slice.buffer,
            .format = format,
            .offset = slice.offset,
            .range = range,
        });
    }
    return *view;
}

void UploadRing::Reset() {
    current_ = 0, used_ = 0;
    if (views_.size() > 4096) views_.clear(); // slices that stopped repeating; the ring's last use has completed
}

void UploadRing::Invalidate() {
    for (size_t i = 0; i <= current_ && i < chunks_.size(); ++i) chunks_[i].Invalidate();
}

VkDescriptorSet DescriptorArena::Allocate(VkDescriptorSetLayout layout) {
    const auto& dev = *Vk().device;
    const VkDescriptorSetAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = VK_NULL_HANDLE,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    for (;;) {
        const bool fresh = current_ == pools_.size();
        if (fresh) {
            // Room for 256 sets of a pipeline over the push limit (32 on NVIDIA); a full pool moves on to the
            // next, kept for the submission's later uses.
            constexpr u32 kSets = 256;
            const std::array<VkDescriptorPoolSize, 6> sizes{{
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSets * 24},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSets * 16},
                {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, kSets * 8},
                {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, kSets * 8},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSets * 32},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSets * 8},
            }};
            pools_.push_back(dev.GetLogical().CreateDescriptorPool({
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .maxSets = kSets,
                .poolSizeCount = static_cast<u32>(sizes.size()),
                .pPoolSizes = sizes.data(),
            }));
        }
        VkDescriptorSetAllocateInfo info = ai;
        info.descriptorPool = *pools_[current_];
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult r = dev.GetDispatchLoader().vkAllocateDescriptorSets(*dev.GetLogical(), &info, &set);
        if (r == VK_SUCCESS) return set;
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) Fatal("nvn: vkAllocateDescriptorSets: VkResult %d", r);
        if (fresh) Fatal("nvn: a descriptor set larger than a whole descriptor pool");
        ++current_;
    }
}

void DescriptorArena::Reset() {
    static const auto reset = reinterpret_cast<PFN_vkResetDescriptorPool>(
        Vk().dld.vkGetDeviceProcAddr(*Vk().device->GetLogical(), "vkResetDescriptorPool"));
    for (size_t i = 0; i < current_ + 1 && i < pools_.size(); ++i) reset(*Vk().device->GetLogical(), *pools_[i], 0);
    current_ = 0;
}

// --- CopyBufferToTexture ---------------------------------------------------------------------------

namespace {
/// NVN's copy geometry between a texture and tightly packed buffer rows (BRO 0x565550, the same for both
/// directions): the Vulkan region (bufferOffset 0) and the buffer's byte size; stops the run for what the
/// layer does not handle.
struct CopyGeometry {
    VkBufferImageCopy region;
    u64 size;
};
CopyGeometry Geometry(const char* fn, const CopyBufferToTextureRecord& r) {
    const TextureBuilder& b = r.texture.settings;
    const TextureView& v = r.view.view;
    if (r.flags) Fatal("nvn: %s with flags 0x%x is not implemented yet", fn, r.flags);
    if (r.view.present && (v.set & kViewFormat) && v.format != b.format) {
        Fatal("nvn: %s through a view of format 0x%x (texture 0x%x) is not implemented yet", fn, v.format, b.format);
    }
    const FormatInfo* f = FindFormat(b.format);
    if (!f) Fatal("nvn: %s: unknown format 0x%x", fn, b.format);
    // Depth or stencil alone (D16, D32F, S8): the buffer bytes are Vulkan's copy layout. Packed depth +
    // stencil (D24S8, D32FS8) would need splitting into aspects.
    if (f->depth_stencil && (b.format == 0x35 || b.format == 0x36)) {
        Fatal("nvn: %s with depth/stencil format 0x%x is not implemented yet", fn, b.format);
    }
    const CopyRegion& g = r.region;
    if (g.x < 0 || g.y < 0 || g.z < 0 || g.width <= 0 || g.height <= 0 || g.depth <= 0) {
        Fatal("nvn: %s: region %d,%d,%d %dx%dx%d", fn, g.x, g.y, g.z, g.width, g.height, g.depth);
    }
    const u32 level = r.view.present && (v.set & kViewLevels) ? v.base_level : 0;
    const u32 base_layer = r.view.present && (v.set & kViewLayers) ? static_cast<u32>(v.base_layer) : 0;

    // NVN's block rounding; the rows of a 1D array are its layers
    const bool array1d = b.target == kTarget1DArray;
    const u32 xb = CeilDiv(g.x, f->block_w), wb = CeilDiv(g.width, f->block_w);
    const u32 yb = array1d ? 0 : CeilDiv(g.y, f->block_h), hb = array1d ? 1 : CeilDiv(g.height, f->block_h);
    const u32 first = array1d ? g.y : g.z, count = array1d ? g.height : g.depth; // layers or 3D slices
    const u64 row = u64{wb} * f->bytes, image = row * hb, size = image * count;

    const u32 lw = std::max(1u, b.width >> level), lh = array1d ? 1 : std::max(1u, b.height >> level);
    const u32 ld = b.target == kTarget3D ? std::max(1u, b.depth >> level) : 1;
    const u32 layers = b.target == kTarget2DArray ? b.depth : b.target == kTargetCube ? 6 : array1d ? b.height : 1;
    const bool sliced = b.target == kTarget3D;
    if (level >= std::max<u32>(b.levels, 1) || xb + wb > CeilDiv(lw, f->block_w) ||
        yb + hb > CeilDiv(lh, f->block_h) || (sliced ? first + count > ld : base_layer + first + count > layers)) {
        Fatal("nvn: %s: region %d,%d,%d %dx%dx%d level %u outside the texture (%ux%u, %u %s)", fn, g.x, g.y, g.z,
              g.width, g.height, g.depth, level, lw, lh, sliced ? ld : layers, sliced ? "slices" : "layers");
    }
    const PoolRange buf = FindPool(r.buffer);
    if (!buf.pool || buf.offset + size > buf.size) {
        Fatal("nvn: %s: 0x%llx bytes at 0x%llx are not in one memory pool", fn, (unsigned long long)size,
              (unsigned long long)r.buffer);
    }
    const u32 x = xb * f->block_w, y = yb * f->block_h;
    const VkImageAspectFlags aspect = b.format == 0x31 ? VK_IMAGE_ASPECT_STENCIL_BIT
                                      : f->depth_stencil ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                         : VK_IMAGE_ASPECT_COLOR_BIT;
    return {VkBufferImageCopy{
                .bufferOffset = 0,
                .bufferRowLength = wb * f->block_w,
                .bufferImageHeight = hb * f->block_h,
                .imageSubresource = {aspect, level, sliced ? 0 : base_layer + first, sliced ? 1 : count},
                .imageOffset = {static_cast<s32>(x), static_cast<s32>(y), static_cast<s32>(sliced ? first : 0)},
                .imageExtent = {std::min(wb * f->block_w, lw - x), std::min(hb * f->block_h, lh - y), sliced ? count : 1},
            },
            size};
}
} // namespace

void TranslateCopyBufferToTexture(Translator& t, const CopyBufferToTextureRecord& r) {
    EndOpenPass(t.cmd);
    CopyGeometry g = Geometry("CopyBufferToTexture", r);
    const auto slice = t.upload.Allocate(g.size, 16);
    g_mem.Read(r.buffer, slice.data, g.size); // 5.0 rule 1: the bytes as they are at submit
    const auto img = ImageFor(r.texture, t.cmd);
    t.keep.push_back(img);
    g.region.bufferOffset = slice.offset;
    if (!img->Scaled()) {
        t.cmd.CopyBufferToImage(slice.buffer, *img->image, VK_IMAGE_LAYOUT_GENERAL, g.region);
    } else { // into an unscaled twin, then blitted up
        if (img->aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
            Fatal("nvn: CopyBufferToTexture into a scaled depth or stencil image (RT_RES) is not implemented yet");
        }
        const auto twin = Twin(r.texture, t.cmd);
        t.keep.push_back(twin);
        t.cmd.CopyBufferToImage(slice.buffer, *twin->image, VK_IMAGE_LAYOUT_GENERAL, g.region);
        FullBarrier(t.cmd);
        BlitArea(t.cmd, *twin, *img, g.region.imageSubresource, g.region.imageOffset, g.region.imageExtent,
                 VK_FILTER_LINEAR);
    }
    NoteImageUse(r.texture, ImageUse::kUpload, g.region.imageSubresource.mipLevel);
    FullBarrier(t.cmd); // copies are ordered on NVN's engines; until the barrier rules are worked out (5.2)
}

// CopyTextureToBuffer(texture, view, region, buffer, flags): the GPU writes guest memory, so the rows go to
// the submission's download ring and reach the buffer when the submission retires (5.0 rules 2-3).
void TranslateCopyTextureToBuffer(Translator& t, const CopyBufferToTextureRecord& r) {
    EndOpenPass(t.cmd);
    CopyGeometry g = Geometry("CopyTextureToBuffer", r);
    const auto slice = t.download->Allocate(g.size, 16);
    const auto img = ImageFor(r.texture, t.cmd);
    t.keep.push_back(img);
    FullBarrier(t.cmd);
    g.region.bufferOffset = slice.offset;
    if (!img->Scaled()) {
        t.cmd.CopyImageToBuffer(*img->image, VK_IMAGE_LAYOUT_GENERAL, slice.buffer, g.region);
    } else { // blitted down into an unscaled twin, read from there
        if (img->aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
            Fatal("nvn: CopyTextureToBuffer from a scaled depth or stencil image (RT_RES) is not implemented yet");
        }
        const auto twin = Twin(r.texture, t.cmd);
        t.keep.push_back(twin);
        BlitArea(t.cmd, *img, *twin, g.region.imageSubresource, g.region.imageOffset, g.region.imageExtent,
                 VK_FILTER_LINEAR);
        FullBarrier(t.cmd);
        t.cmd.CopyImageToBuffer(*twin->image, VK_IMAGE_LAYOUT_GENERAL, slice.buffer, g.region);
    }
    NoteImageUse(r.texture, ImageUse::kToBuffer, g.region.imageSubresource.mipLevel);
    FullBarrier(t.cmd);
    t.writebacks->push_back({WriteBack::Kind::kRing, r.buffer, g.size, slice.data, {}, 0, slice.buffer, slice.offset});
    t.writebacks->back().origin = WriteBack::Origin::kCopyTexture;
}

// --- CopyTextureToTexture ---------------------------------------------------------------------------
// NVN copies between textures with its 2D engine, which scales: regions of the same size and format are a
// vkCmdCopyImage, others a vkCmdBlitImage (linear filtering with flag bit 0, NVN's LINEAR_FILTER). Not yet:
// mirroring (flags 4/8/16), blits of compressed or depth formats.

namespace {
struct TextureArea {
    u32 level, first_layer, layers; // layers: array layers (or 1 for 3D)
    VkOffset3D offset;
    VkExtent3D extent;
};
TextureArea AreaOf(const TextureState& t, const ViewArg& va, const CopyRegion& g) {
    const TextureBuilder& b = t.settings;
    const TextureView& v = va.view;
    if (va.present && (v.set & kViewFormat) && v.format != b.format) {
        Fatal("nvn: CopyTextureToTexture through a view of format 0x%x (texture 0x%x) is not implemented yet", v.format, b.format);
    }
    if (g.x < 0 || g.y < 0 || g.z < 0 || g.width <= 0 || g.height <= 0 || g.depth <= 0) {
        Fatal("nvn: CopyTextureToTexture: region %d,%d,%d %dx%dx%d", g.x, g.y, g.z, g.width, g.height, g.depth);
    }
    const u32 level = va.present && (v.set & kViewLevels) ? v.base_level : 0;
    const u32 base = va.present && (v.set & kViewLayers) ? static_cast<u32>(v.base_layer) : 0;
    const bool array1d = b.target == kTarget1DArray, sliced = b.target == kTarget3D;
    TextureArea a{level, 0, 1, {g.x, array1d ? 0 : g.y, 0}, {static_cast<u32>(g.width), array1d ? 1u : static_cast<u32>(g.height), 1}};
    if (sliced) a.offset.z = g.z, a.extent.depth = static_cast<u32>(g.depth);
    else a.first_layer = base + static_cast<u32>(array1d ? g.y : g.z), a.layers = static_cast<u32>(array1d ? g.height : g.depth);
    const u32 lw = std::max(1u, b.width >> level), lh = array1d ? 1 : std::max(1u, b.height >> level);
    if (level >= std::max<u32>(b.levels, 1) || a.offset.x + a.extent.width > lw || a.offset.y + a.extent.height > lh) {
        Fatal("nvn: CopyTextureToTexture: region %d,%d %dx%d outside level %u (%ux%u)", g.x, g.y, g.width, g.height, level, lw, lh);
    }
    return a;
}
} // namespace

void TranslateCopyTextureToTexture(Translator& t, const CopyTextureToTextureRecord& r) {
    EndOpenPass(t.cmd);
    if (r.flags & ~3u) Fatal("nvn: CopyTextureToTexture with flags 0x%x is not implemented yet", r.flags);
    const TextureArea s = AreaOf(r.src, r.src_view, r.src_region), d = AreaOf(r.dst, r.dst_view, r.dst_region);
    if (s.layers != d.layers) Fatal("nvn: CopyTextureToTexture between %u and %u layers", s.layers, d.layers);
    const auto si = ImageFor(r.src, t.cmd), di = ImageFor(r.dst, t.cmd);
    t.keep.push_back(si);
    t.keep.push_back(di);
    NoteImageUse(r.src, ImageUse::kCopySrc, s.level), NoteImageUse(r.dst, ImageUse::kCopyDst, d.level);
    FullBarrier(t.cmd);
    const VkImageSubresourceLayers ss{si->aspect, s.level, s.first_layer, s.layers}, ds{di->aspect, d.level, d.first_layer, d.layers};
    if (si->Scaled() || di->Scaled()) { // each region in its image's own units
        const auto sc = Corners(*si, s.offset, s.extent), dc = Corners(*di, d.offset, d.extent);
        const VkExtent3D se{static_cast<u32>(sc[1].x - sc[0].x), static_cast<u32>(sc[1].y - sc[0].y), s.extent.depth};
        const VkExtent3D de{static_cast<u32>(dc[1].x - dc[0].x), static_cast<u32>(dc[1].y - dc[0].y), d.extent.depth};
        const auto& dld = Vk().device->GetDispatchLoader();
        if (si->format == di->format && se.width == de.width && se.height == de.height && se.depth == de.depth) {
            const VkImageCopy copy{ss, sc[0], ds, dc[0], se};
            dld.vkCmdCopyImage(*t.cmd, *si->image, VK_IMAGE_LAYOUT_GENERAL, *di->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
        } else {
            if (si->aspect != VK_IMAGE_ASPECT_COLOR_BIT || di->aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
                Fatal("nvn: CopyTextureToTexture between depth images of different scales (RT_RES) is not implemented yet");
            }
            const VkImageBlit blit{ss, {sc[0], sc[1]}, ds, {dc[0], dc[1]}};
            const bool linear = (r.flags & 1) || si->scale_n != di->scale_n;
            dld.vkCmdBlitImage(*t.cmd, *si->image, VK_IMAGE_LAYOUT_GENERAL, *di->image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                               linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
        }
        FullBarrier(t.cmd);
        return;
    }
    const bool same = si->format == di->format && s.extent.width == d.extent.width && s.extent.height == d.extent.height &&
                      s.extent.depth == d.extent.depth;
    const auto& dld = Vk().device->GetDispatchLoader();
    if (same) {
        const VkImageCopy copy{ss, s.offset, ds, d.offset, s.extent};
        dld.vkCmdCopyImage(*t.cmd, *si->image, VK_IMAGE_LAYOUT_GENERAL, *di->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    } else {
        const FormatInfo* sf = FindFormat(r.src.settings.format);
        const FormatInfo* df = FindFormat(r.dst.settings.format);
        if (!sf || !df || sf->depth_stencil || df->depth_stencil || sf->block_w > 1 || df->block_w > 1) {
            Fatal("nvn: CopyTextureToTexture scaling or converting format 0x%x to 0x%x is not implemented yet",
                  r.src.settings.format, r.dst.settings.format);
        }
        auto end = [](VkOffset3D o, VkExtent3D e) {
            return VkOffset3D{o.x + static_cast<s32>(e.width), o.y + static_cast<s32>(e.height), o.z + static_cast<s32>(e.depth)};
        };
        const VkImageBlit blit{ss, {s.offset, end(s.offset, s.extent)}, ds, {d.offset, end(d.offset, d.extent)}};
        dld.vkCmdBlitImage(*t.cmd, *si->image, VK_IMAGE_LAYOUT_GENERAL, *di->image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                           (r.flags & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    FullBarrier(t.cmd);
}

// --- ClearTexture ----------------------------------------------------------------------------------
// vkCmdClearColorImage clears whole levels and layers: a region that covers
// the level's width and height (and all slices of a 3D level) is translated;
// anything smaller, a channel mask other than all four, and depth/stencil
// textures stop the run until a game needs them.

void TranslateClearTexture(Translator& t, const ClearTextureRecord& r) {
    EndOpenPass(t.cmd);
    const TextureBuilder& b = r.texture.settings;
    const TextureView& v = r.view.view;
    const FormatInfo* f = FindFormat(b.format);
    if (!f) Fatal("nvn: ClearTexture: unknown format 0x%x", b.format);
    if (f->depth_stencil) Fatal("nvn: ClearTexture of depth/stencil format 0x%x is not implemented yet", b.format);
    if ((r.mask & 0xF) != 0xF) Fatal("nvn: ClearTexture with channel mask 0x%x is not implemented yet", r.mask);
    if (r.view.present && (v.set & kViewFormat) && v.format != b.format) {
        Fatal("nvn: ClearTexture through a view of format 0x%x (texture 0x%x) is not implemented yet", v.format, b.format);
    }
    const CopyRegion& g = r.region;
    const u32 level = r.view.present && (v.set & kViewLevels) ? v.base_level : 0;
    const u32 base_layer = r.view.present && (v.set & kViewLayers) ? static_cast<u32>(v.base_layer) : 0;
    const bool array1d = b.target == kTarget1DArray, sliced = b.target == kTarget3D;
    const u32 lw = std::max(1u, b.width >> level), lh = array1d ? 1 : std::max(1u, b.height >> level);
    const u32 ld = sliced ? std::max(1u, b.depth >> level) : 1;
    const s32 first = array1d ? g.y : g.z, count = array1d ? g.height : g.depth;
    const bool whole = g.x == 0 && g.width >= static_cast<s32>(lw) && (array1d || (g.y == 0 && g.height >= static_cast<s32>(lh))) &&
                       (!sliced || (g.z == 0 && g.depth >= static_cast<s32>(ld)));
    if (!whole || first < 0 || count <= 0) {
        Fatal("nvn: ClearTexture of region %d,%d,%d %dx%dx%d (level %u: %ux%ux%u) is not implemented yet", g.x, g.y,
              g.z, g.width, g.height, g.depth, level, lw, lh, ld);
    }
    const auto img = ImageFor(r.texture, t.cmd);
    t.keep.push_back(img);
    VkClearColorValue color;
    std::memcpy(color.uint32, r.color, sizeof color.uint32);
    const VkImageSubresourceRange range{img->aspect, level, 1, sliced ? 0 : base_layer + static_cast<u32>(first),
                                        sliced ? 1 : static_cast<u32>(count)};
    // through the dispatch table: drippu's wrapper has ClearColorImage as a non-const member
    Vk().device->GetDispatchLoader().vkCmdClearColorImage(*t.cmd, *img->image, VK_IMAGE_LAYOUT_GENERAL, &color, 1, &range);
    NoteImageUse(r.texture, ImageUse::kClear, level);
    FullBarrier(t.cmd);
}

// --- RT_NVN_LOG_TARGETS ----------------------------------------------------------------------------

namespace {
bool LogTargetsOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_NVN_LOG_TARGETS");
        return e && *e && *e != '0';
    }();
    return on;
}

struct TargetUses {
    TextureBuilder b;
    double first = 0, last = 0;
    std::array<u64, static_cast<size_t>(ImageUse::kCount)> uses{};
    u32 drawn_levels = 0; // bit per level used as a render target
    float vp_min_w = 0, vp_min_h = 0, vp_max_w = 0, vp_max_h = 0;
};
std::mutex g_target_uses_mu;
std::map<std::array<u64, 4>, TargetUses> g_target_uses; // by (address, format << 32 | target, width << 32 | height, depth << 32 | levels)
double g_target_uses_written = 0;

std::array<u64, 4> TargetKey(const TextureState& t) {
    const TextureBuilder& b = t.settings;
    return {t.address, u64{b.format} << 32 | b.target, u64{b.width} << 32 | b.height, u64{b.depth} << 32 | b.levels};
}

void WriteTargetUses() { // with g_target_uses_mu
    FILE* f = std::fopen("targets.csv", "w");
    if (!f) return;
    std::fprintf(f, "address,format,target,width,height,depth,levels,first_s,last_s,color,depth_target,clear,sampled,"
                    "storage,copy_src,copy_dst,upload,to_buffer,present,capture,drawn_levels,vp_min_w,vp_min_h,vp_max_w,"
                    "vp_max_h\n");
    for (const auto& [key, u] : g_target_uses) {
        std::fprintf(f, "0x%llx,0x%x,%u,%u,%u,%u,%u,%.1f,%.1f", (unsigned long long)key[0], u.b.format, u.b.target,
                     u.b.width, u.b.height, u.b.depth, u.b.levels, u.first, u.last);
        for (u64 n : u.uses) std::fprintf(f, ",%llu", (unsigned long long)n);
        std::fprintf(f, ",0x%x,%.0f,%.0f,%.0f,%.0f\n", u.drawn_levels, u.vp_min_w, u.vp_min_h, u.vp_max_w, u.vp_max_h);
    }
    std::fclose(f);
}

TargetUses& UsesOf(const TextureState& t, double now) { // with g_target_uses_mu
    TargetUses& u = g_target_uses[TargetKey(t)];
    if (!u.first) u.b = t.settings, u.first = now;
    u.last = now;
    if (now >= g_target_uses_written + 5) g_target_uses_written = now, WriteTargetUses(); // the run usually ends by being killed
    return u;
}
} // namespace

void NoteImageUse(const TextureState& t, ImageUse use, u32 level) {
    if (!LogTargetsOn()) return;
    std::lock_guard lock(g_target_uses_mu);
    TargetUses& u = UsesOf(t, Uptime());
    ++u.uses[static_cast<size_t>(use)];
    if (use == ImageUse::kColor || use == ImageUse::kDepth) u.drawn_levels |= 1u << std::min(level, 31u);
}

void NoteTargetViewport(const TextureState& t, float width, float height) {
    if (!LogTargetsOn()) return;
    std::lock_guard lock(g_target_uses_mu);
    TargetUses& u = UsesOf(t, Uptime());
    if (!u.vp_max_w || width < u.vp_min_w) u.vp_min_w = width;
    if (!u.vp_max_h || height < u.vp_min_h) u.vp_min_h = height;
    u.vp_max_w = std::max(u.vp_max_w, width), u.vp_max_h = std::max(u.vp_max_h, height);
}

} // namespace rt::nvn
