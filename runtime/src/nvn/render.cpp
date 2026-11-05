// Render state and clears.
//
// Recording (signatures checked in BRO, 2026-10-16): SetRenderTargets(count,
// colours, colour views, depth, depth view) — NVN reads the textures and views
// when it records, so the records keep copies; SetViewport / SetScissor /
// SetDepthRange are SetViewports / SetScissors / SetDepthRanges with first 0,
// count 1 (BRO 0x585e50 ...); BindMultisampleState copies the state object;
// ClearColor(index, colour[4], channel mask); ClearDepthStencil(depth, depth
// mask, stencil, stencil mask). SaveZCullData / RestoreZCullData keep the
// Maxwell GPU's ZCull memory, which Vulkan has no use for: nothing recorded.
//
// Translation: render-state records update the queue's RenderState (render.h)
// in submission order; clears run in a render pass of the one target they
// clear, within the scissor of viewport 0 when one was set (inferred: NVN's
// clears follow the scissor, as GL's do; to check against the driver seam's
// captures), else the whole target. Not yet: colour channel masks other than
// all four, stencil masks other than 0 or 0xff (Vulkan clears ignore write
// masks), layered clears beyond the view's layers.
#include "pipeline.h"

#include <cstring>
#include <map>
#include <mutex>

namespace rt::nvn {

namespace {

struct TargetsHeader {
    u32 count, has_depth;
};
struct RangeHeader {
    u32 first, count;
};
struct ClearColorRecord {
    s32 index;
    u32 color[4]; // float, int or uint bits, as passed
    u32 mask;
};
struct ClearDepthStencilRecord {
    float depth;
    u32 depth_mask;
    s32 stencil;
    u32 stencil_mask;
};

RenderTarget ReadTarget(u64 texture, u64 view) {
    RenderTarget r{};
    if (!texture) return r; // an unbound slot: address 0
    r.texture = TextureStateOf(texture);
    r.view = ReadView(view);
    return r;
}

/// Records a header and `count` entries of `entry_size` bytes from guest memory.
void RecordRange(GuestContext* c, u64 cmdbuf, Op op, s32 first, s32 count, u64 entries, u32 entry_size) {
    if (first < 0 || count < 0 || first + count > static_cast<s32>(kMaxViewports)) {
        Fatal("nvn: %s(first %d, count %d)", OpName(op), first, count);
    }
    std::vector<u8> payload(sizeof(RangeHeader) + entry_size * static_cast<u32>(count));
    const RangeHeader h{static_cast<u32>(first), static_cast<u32>(count)};
    std::memcpy(payload.data(), &h, sizeof h);
    g_mem.Read(entries, payload.data() + sizeof h, payload.size() - sizeof h);
    Record(c, cmdbuf, op, payload.data(), static_cast<u32>(payload.size()));
}

} // namespace

// --- Recording ------------------------------------------------------------------------------------

NVN_PROC(nvnCommandBufferSetRenderTargets, void, u64 cmdbuf, s32 count, u64 colors, u64 views, u64 depth,
         u64 depth_view) {
    if (count < 0 || count > static_cast<s32>(kMaxColorTargets)) Fatal("nvn: SetRenderTargets with %d targets", count);
    std::vector<u8> payload(sizeof(TargetsHeader) + sizeof(RenderTarget) * (1 + static_cast<u32>(count)));
    const TargetsHeader h{static_cast<u32>(count), depth != 0};
    std::memcpy(payload.data(), &h, sizeof h);
    u8* at = payload.data() + sizeof h;
    const RenderTarget d = ReadTarget(depth, depth_view);
    std::memcpy(at, &d, sizeof d);
    for (s32 i = 0; i < count; ++i) {
        const u64 tex = colors ? hle::ReadGuest<u64>(colors + 8ULL * i) : 0;
        const u64 view = views ? hle::ReadGuest<u64>(views + 8ULL * i) : 0;
        const RenderTarget t = ReadTarget(tex, view);
        std::memcpy(at + sizeof(RenderTarget) * (1 + i), &t, sizeof t);
    }
    Record(c, cmdbuf, Op::kSetRenderTargets, payload.data(), static_cast<u32>(payload.size()));
}

NVN_PROC(nvnCommandBufferSetViewports, void, u64 cmdbuf, s32 first, s32 count, u64 rects) {
    RecordRange(c, cmdbuf, Op::kSetViewports, first, count, rects, 16);
}
NVN_PROC(nvnCommandBufferSetViewport, void, u64 cmdbuf, s32 x, s32 y, s32 w, s32 h) {
    const float v[4] = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(w), static_cast<float>(h)};
    u8 payload[sizeof(RangeHeader) + 16];
    const RangeHeader hd{0, 1};
    std::memcpy(payload, &hd, sizeof hd);
    std::memcpy(payload + sizeof hd, v, 16);
    Record(c, cmdbuf, Op::kSetViewports, payload, sizeof payload);
}
NVN_PROC(nvnCommandBufferSetDepthRanges, void, u64 cmdbuf, s32 first, s32 count, u64 ranges) {
    RecordRange(c, cmdbuf, Op::kSetDepthRanges, first, count, ranges, 8);
}
NVN_PROC(nvnCommandBufferSetDepthRange, void, u64 cmdbuf, float n, float f) {
    u8 payload[sizeof(RangeHeader) + 8];
    const RangeHeader hd{0, 1};
    std::memcpy(payload, &hd, sizeof hd);
    std::memcpy(payload + sizeof hd, &n, 4);
    std::memcpy(payload + sizeof hd + 4, &f, 4);
    Record(c, cmdbuf, Op::kSetDepthRanges, payload, sizeof payload);
}
NVN_PROC(nvnCommandBufferSetScissors, void, u64 cmdbuf, s32 first, s32 count, u64 rects) {
    RecordRange(c, cmdbuf, Op::kSetScissors, first, count, rects, 16);
}
NVN_PROC(nvnCommandBufferSetScissor, void, u64 cmdbuf, s32 x, s32 y, s32 w, s32 h) {
    u8 payload[sizeof(RangeHeader) + 16];
    const RangeHeader hd{0, 1};
    const s32 r[4] = {x, y, w, h};
    std::memcpy(payload, &hd, sizeof hd);
    std::memcpy(payload + sizeof hd, r, 16);
    Record(c, cmdbuf, Op::kSetScissors, payload, sizeof payload);
}
NVN_PROC(nvnCommandBufferBindMultisampleState, void, u64 cmdbuf, u64 state) {
    Record(c, cmdbuf, Op::kBindMultisampleState, hle::ReadGuest<std::array<u8, 24>>(state));
}

#define NVN_CLEAR_COLOR(fn)                                                                                         \
    NVN_PROC(fn, void, u64 cmdbuf, s32 index, u64 color, u32 mask) {                                                \
        ClearColorRecord r{};                                                                                       \
        r.index = index;                                                                                            \
        const auto words = hle::ReadGuest<std::array<u32, 4>>(color);                                               \
        std::memcpy(r.color, words.data(), sizeof r.color);                                                         \
        r.mask = mask;                                                                                              \
        Record(c, cmdbuf, Op::kClearColor, r);                                                                      \
    }
NVN_CLEAR_COLOR(nvnCommandBufferClearColor)
NVN_CLEAR_COLOR(nvnCommandBufferClearColori)
NVN_CLEAR_COLOR(nvnCommandBufferClearColorui)
#undef NVN_CLEAR_COLOR

NVN_PROC(nvnCommandBufferClearDepthStencil, void, u64 cmdbuf, float depth, bool depth_mask, s32 stencil,
         s32 stencil_mask) {
    Record(c, cmdbuf, Op::kClearDepthStencil,
           ClearDepthStencilRecord{depth, depth_mask ? 1u : 0u, stencil, static_cast<u32>(stencil_mask)});
}

// SetShaderScratchMemory(pool, offset, size): memory Maxwell shaders spill to; Vulkan drivers spill on their
// own. A control command in NVN (kind 16, 16 bytes; BRO 0x56dc20), so a control record of that size here.
NVN_PROC(nvnCommandBufferSetShaderScratchMemory, void, u64 cmdbuf, u64 pool, s64 offset, u64 size) {
    ControlRecord(c, cmdbuf, Op::kScratchMemory, size, 16);
}
NVN_PROC(nvnCommandBufferSetDepthBounds, void, u64 cmdbuf, bool enable, float min, float max) {
    struct { u32 enable; float min, max; } r{enable ? 1u : 0u, min, max};
    Record(c, cmdbuf, Op::kSetDepthBounds, r);
}
// Debug groups and markers: tools only.
NVN_PROC(nvnCommandBufferPushDebugGroup, void, u64 cmdbuf, u64 name) {}
NVN_PROC(nvnCommandBufferPushDebugGroupStatic, void, u64 cmdbuf, u32 id, u64 name) {}
NVN_PROC(nvnCommandBufferPushDebugGroupDynamic, void, u64 cmdbuf, u32 id, u64 name) {}
NVN_PROC(nvnCommandBufferPopDebugGroup, void, u64 cmdbuf) {}
NVN_PROC(nvnCommandBufferPopDebugGroupId, void, u64 cmdbuf, u32 id) {}
NVN_PROC(nvnCommandBufferInsertDebugMarker, void, u64 cmdbuf, u64 name) {}
NVN_PROC(nvnCommandBufferInsertDebugMarkerStatic, void, u64 cmdbuf, u32 id, u64 name) {}
NVN_PROC(nvnCommandBufferInsertDebugMarkerDynamic, void, u64 cmdbuf, u32 id, u64 name) {}

// SetTiledCacheAction: a hint for Maxwell's tiled caching (binning) of render-target writes; Vulkan has none.
NVN_PROC(nvnCommandBufferSetTiledCacheAction, void, u64 cmdbuf, u32 action) {}
NVN_PROC(nvnCommandBufferSaveZCullData, void, u64 cmdbuf, u64 address, u64 size) {}
NVN_PROC(nvnCommandBufferRestoreZCullData, void, u64 cmdbuf, u64 address, u64 size) {}

// --- Translation ----------------------------------------------------------------------------------

namespace {

std::mutex g_passes_mu;
std::mutex g_attachments_mu; // TextureImage::attachments of every image
std::map<std::pair<VkFormat, bool>, Vulkan::vk::RenderPass> g_passes; // (format, depth) -> clear pass

VkRenderPass ClearPass(VkFormat format, bool depth) {
    std::lock_guard lock(g_passes_mu);
    auto& pass = g_passes[{format, depth}];
    if (!pass) {
        const VkAttachmentDescription a{
            .flags = 0,
            .format = format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
            .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
            .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_GENERAL};
        const VkSubpassDescription sub{
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = depth ? 0u : 1u,
            .pColorAttachments = depth ? nullptr : &ref,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = depth ? &ref : nullptr,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };
        pass = Vk().device->GetLogical().CreateRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .attachmentCount = 1,
            .pAttachments = &a,
            .subpassCount = 1,
            .pSubpasses = &sub,
            .dependencyCount = 0,
            .pDependencies = nullptr,
        });
    }
    return *pass;
}

/// The level of a render target's view.
u32 TargetLevel(const RenderTarget& rt) {
    return rt.view.present && (rt.view.view.set & kViewLevels) ? rt.view.view.base_level : 0;
}

/// The attachment view of a render target (level and layers from its view), created at first use.
const TextureImage::Attachment& AttachmentOf(TextureImage& img, const RenderTarget& rt, bool depth) {
    const TextureBuilder& b = rt.texture.settings;
    const TextureView& v = rt.view.view;
    if (rt.view.present && (v.set & kViewFormat) && v.format != b.format) {
        Fatal("nvn: render target through a view of format 0x%x (texture 0x%x) is not implemented yet", v.format, b.format);
    }
    const u32 level = TargetLevel(rt);
    // 3D: the depth slices of the level, attached as a 2D array (the image is 2D-array compatible, images.cpp)
    const u32 all_layers = b.target == 2 ? std::max(1u, b.depth >> level) : ArrayLayers(b);
    u32 base = 0, layers = all_layers;
    if (rt.view.present && (v.set & kViewLayers)) {
        base = static_cast<u32>(v.base_layer);
        // a layer count of 0 (Brothership sets it on 2D targets): the layers from base on (inferred)
        layers = v.num_layers > 0 ? static_cast<u32>(v.num_layers) : all_layers - std::min(base, all_layers);
    }
    if (layers == 0 || base + layers > all_layers) Fatal("nvn: render target layers %u+%u of %u", base, layers, all_layers);
    std::lock_guard lock(g_attachments_mu);
    auto& a = img.attachments[{level, base, layers}];
    if (!a.view) {
        const auto& logical = Vk().device->GetLogical();
        a.view = logical.CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *img.image,
            .viewType = b.target == 0 || b.target == 3 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = img.format,
            .components = {},
            .subresourceRange = {img.aspect, level, 1, base, layers},
        });
        a.extent = {std::max(1u, b.width >> level), b.target == 0 || b.target == 3 ? 1u : std::max(1u, b.height >> level)};
        if (img.Scaled()) a.extent = {img.UpSize(a.extent.width), img.UpSize(a.extent.height)}; // scaled targets
        a.layers = layers;
        const VkImageView view = *a.view;
        a.framebuffer = logical.CreateFramebuffer({
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderPass = ClearPass(img.format, depth),
            .attachmentCount = 1,
            .pAttachments = &view,
            .width = a.extent.width,
            .height = a.extent.height,
            .layers = layers,
        });
    }
    return a;
}

/// Clears `aspects` of one render target within the scissor of viewport 0 (if set).
void ClearTarget(Translator& t, const RenderTarget& rt, bool depth, VkImageAspectFlags aspects, const VkClearValue& value) {
    const auto img = ImageFor(rt.texture, t.cmd);
    t.keep.push_back(img);
    const TextureImage::Attachment& a = AttachmentOf(*img, rt, depth);
    NoteImageUse(rt.texture, ImageUse::kClear, TargetLevel(rt));
    s32 x0 = 0, y0 = 0, x1 = static_cast<s32>(a.extent.width), y1 = static_cast<s32>(a.extent.height);
    if (t.render->scissor_set[0]) { // (the game's units; a scaled image's own: Up)
        const auto& s = t.render->scissors[0];
        x0 = std::max(x0, img->Up(s[0])), y0 = std::max(y0, img->Up(s[1]));
        x1 = std::min(x1, img->Up(s[0] + s[2])), y1 = std::min(y1, img->Up(s[1] + s[3]));
    }
    if (x1 <= x0 || y1 <= y0) return; // nothing inside the scissor
    const auto& dld = Vk().device->GetDispatchLoader();
    EndOpenPass(t.cmd);
    const VkRenderPassBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = ClearPass(img->format, depth),
        .framebuffer = *a.framebuffer,
        .renderArea = {{0, 0}, a.extent},
        .clearValueCount = 0,
        .pClearValues = nullptr,
    };
    dld.vkCmdBeginRenderPass(*t.cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkClearAttachment clear{aspects, 0, value};
    const VkClearRect rect{{{x0, y0}, {static_cast<u32>(x1 - x0), static_cast<u32>(y1 - y0)}}, 0, a.layers};
    dld.vkCmdClearAttachments(*t.cmd, 1, &clear, 1, &rect);
    dld.vkCmdEndRenderPass(*t.cmd);
    FullBarrier(t.cmd);
}

std::mutex g_targets_mu;
std::map<std::vector<VkFormat>, Vulkan::vk::RenderPass> g_target_passes;
struct Framebuffer {
    Vulkan::vk::Framebuffer framebuffer;
    std::vector<std::weak_ptr<TextureImage>> images; // a view handle may be reused once its image is gone
};
std::map<std::vector<u64>, Framebuffer> g_framebuffers; // (pass, views...) -> framebuffer

template <typename T>
T Get(const u8* p, size_t size, size_t at = 0) {
    if (at + sizeof(T) > size) Fatal("nvn: a render record is too short");
    T v;
    std::memcpy(&v, p + at, sizeof v);
    return v;
}

} // namespace

bool TranslateRenderRecord(Translator& t, Op op, const u8* p, size_t size) {
    RenderState& rs = *t.render;
    switch (op) {
    case Op::kSetRenderTargets: {
        const auto h = Get<TargetsHeader>(p, size);
        rs.color_count = h.count;
        rs.has_depth = h.has_depth;
        rs.depth = Get<RenderTarget>(p, size, sizeof h);
        rs.colors = {};
        for (u32 i = 0; i < h.count; ++i) rs.colors[i] = Get<RenderTarget>(p, size, sizeof h + sizeof(RenderTarget) * (1 + i));
        return true;
    }
    case Op::kSetViewports:
    case Op::kSetDepthRanges:
    case Op::kSetScissors: {
        const auto h = Get<RangeHeader>(p, size);
        for (u32 i = 0; i < h.count; ++i) {
            const u32 k = h.first + i;
            if (op == Op::kSetViewports) rs.viewports[k] = Get<std::array<float, 4>>(p, size, sizeof h + 16 * i);
            if (op == Op::kSetDepthRanges) rs.depth_ranges[k] = Get<std::array<float, 2>>(p, size, sizeof h + 8 * i);
            if (op == Op::kSetScissors) rs.scissors[k] = Get<std::array<s32, 4>>(p, size, sizeof h + 16 * i), rs.scissor_set[k] = true;
        }
        return true;
    }
    case Op::kScratchMemory: return true;
    case Op::kSetDepthBounds: {
        const auto v = Get<std::array<u32, 3>>(p, size);
        rs.depth_bounds_enable = v[0] != 0;
        std::memcpy(rs.depth_bounds.data(), &v[1], 8);
        return true;
    }
    case Op::kBindMultisampleState:
        rs.multisample = Get<std::array<u8, 24>>(p, size);
        rs.multisample_set = true;
        return true;
    case Op::kClearColor: {
        const auto r = Get<ClearColorRecord>(p, size);
        if (r.index < 0 || r.index >= static_cast<s32>(rs.color_count) || !rs.colors[r.index].texture.address) {
            Fatal("nvn: ClearColor of colour target %d, which is not bound", r.index);
        }
        if ((r.mask & 0xF) != 0xF) Fatal("nvn: ClearColor with channel mask 0x%x is not implemented yet", r.mask);
        VkClearValue v{};
        std::memcpy(v.color.uint32, r.color, sizeof r.color);
        ClearTarget(t, rs.colors[r.index], false, VK_IMAGE_ASPECT_COLOR_BIT, v);
        return true;
    }
    case Op::kClearDepthStencil: {
        const auto r = Get<ClearDepthStencilRecord>(p, size);
        if (!rs.has_depth || !rs.depth.texture.address) Fatal("nvn: ClearDepthStencil without a depth target");
        if (r.stencil_mask != 0 && (r.stencil_mask & 0xFF) != 0xFF) {
            Fatal("nvn: ClearDepthStencil with stencil mask 0x%x is not implemented yet", r.stencil_mask);
        }
        const auto img = ImageFor(rs.depth.texture, t.cmd);
        VkImageAspectFlags aspects = 0;
        if (r.depth_mask) aspects |= img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT;
        if (r.stencil_mask) aspects |= img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT;
        if (!aspects) return true;
        VkClearValue v{};
        v.depthStencil = {r.depth, static_cast<u32>(r.stencil) & 0xFF};
        ClearTarget(t, rs.depth, true, aspects, v);
        return true;
    }
    default: return false;
    }
}

VkRenderPass TargetsPass(const std::array<VkFormat, kMaxColorTargets>& colors, u32 color_count, VkFormat depth) {
    std::vector<VkFormat> key(colors.begin(), colors.begin() + color_count);
    key.push_back(depth);
    std::lock_guard lock(g_targets_mu);
    auto& pass = g_target_passes[key];
    if (!pass) {
        std::vector<VkAttachmentDescription> atts;
        std::vector<VkAttachmentReference> refs;
        auto add = [&](VkFormat f) -> VkAttachmentReference {
            if (f == VK_FORMAT_UNDEFINED) return {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED};
            atts.push_back({
                .flags = 0,
                .format = f,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
                .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
                .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
            });
            return {static_cast<u32>(atts.size() - 1), VK_IMAGE_LAYOUT_GENERAL};
        };
        for (u32 i = 0; i < color_count; ++i) refs.push_back(add(colors[i]));
        const VkAttachmentReference depth_ref = add(depth);
        const VkSubpassDescription sub{
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = color_count,
            .pColorAttachments = refs.data(),
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = depth == VK_FORMAT_UNDEFINED ? nullptr : &depth_ref,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };
        pass = Vk().device->GetLogical().CreateRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .attachmentCount = static_cast<u32>(atts.size()),
            .pAttachments = atts.data(),
            .subpassCount = 1,
            .pSubpasses = &sub,
            .dependencyCount = 0,
            .pDependencies = nullptr,
        });
    }
    return *pass;
}

TargetSet BindTargets(Translator& t) {
    const RenderState& rs = *t.render;
    TargetSet out{};
    out.color_count = rs.color_count;
    out.colors.fill(VK_FORMAT_UNDEFINED);
    out.depth = VK_FORMAT_UNDEFINED;
    std::vector<VkImageView> views;
    std::vector<std::weak_ptr<TextureImage>> images;
    VkExtent2D extent{~0u, ~0u};
    u32 layers = ~0u;
    out.scale_n = 0;
    auto add = [&](const RenderTarget& rt, bool depth) -> VkFormat {
        const TextureBuilder& b = rt.texture.settings;
        if (rt.view.present && (rt.view.view.set & kViewFormat) && rt.view.view.format != b.format) {
            throw NotYet("render targets through a view of another format");
        }
        const auto img = ImageFor(rt.texture, t.cmd);
        t.keep.push_back(img);
        if (out.scale_n && out.scale_n != img->scale_n) throw NotYet("render targets of different scales (RT_RES)");
        out.scale_n = img->scale_n;
        const TextureImage::Attachment& a = AttachmentOf(*img, rt, depth);
        NoteImageUse(rt.texture, depth ? ImageUse::kDepth : ImageUse::kColor, TargetLevel(rt));
        views.push_back(*a.view);
        images.push_back(img);
        extent = {std::min(extent.width, a.extent.width), std::min(extent.height, a.extent.height)};
        layers = std::min(layers, a.layers);
        return img->format;
    };
    for (u32 i = 0; i < rs.color_count; ++i) {
        if (rs.colors[i].texture.address) out.colors[i] = add(rs.colors[i], false);
    }
    if (rs.has_depth && rs.depth.texture.address) out.depth = add(rs.depth, true);
    if (views.empty()) Fatal("nvn: a draw without render targets is not implemented yet");
    out.pass = TargetsPass(out.colors, out.color_count, out.depth);
    out.extent = extent;
    std::vector<u64> key{reinterpret_cast<u64>(out.pass), layers};
    for (VkImageView v : views) key.push_back(reinterpret_cast<u64>(v));
    std::lock_guard lock(g_targets_mu);
    Framebuffer& fb = g_framebuffers[key];
    bool stale = !fb.framebuffer;
    for (const auto& w : fb.images) stale |= w.expired();
    if (stale) {
        fb.framebuffer = Vk().device->GetLogical().CreateFramebuffer({
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderPass = out.pass,
            .attachmentCount = static_cast<u32>(views.size()),
            .pAttachments = views.data(),
            .width = extent.width,
            .height = extent.height,
            .layers = layers,
        });
        fb.images = std::move(images);
    }
    out.framebuffer = *fb.framebuffer;
    return out;
}

} // namespace rt::nvn
