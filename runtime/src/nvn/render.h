// Render state and the commands that set or use it:
// render targets, viewports, depth ranges, scissors, multisample state, clears.
#pragma once

#include "images.h"
#include "program.h"

#include <array>
#include <cstring>

namespace rt::nvn {

constexpr u32 kMaxColorTargets = 8, kMaxViewports = 16;

/// The device's clip modes (device.cpp): NVN_DEPTH_MODE (0 near is -w, 1 near is 0) and
/// NVN_WINDOW_ORIGIN_MODE (0 lower left, 1 upper left).
struct ClipModes {
    u32 depth_mode, window_origin;
};
ClipModes DeviceClipModes(u64 device);

// The driver's constant buffers as NVN lays them out ("Known before Phase 7",
// measured on the running driver): one shared c0 of 0x7000 bytes; per hardware graphics stage a c2 of
// 0x700 bytes holding the bound handles. The layer keeps the game's NVN handles in c2 (the driver writes
// its hardware descriptor numbers there; only the layer reads them back, when it resolves a binding).
constexpr u32 kC0Size = 0x7000, kC2Size = 0x700, kHwStages = 5;
constexpr u32 kC0StorageBuffers = 0x110, kC0BaseVertex = 0x640, kC0BaseInstance = 0x644;
constexpr u32 kC2Textures = 0x20, kC2Images = 0x120, kC2SeparateTextures = 0x168, kC2SeparateSamplers = 0x568;

/// Compute's c0 (report §8.1): 0x500 bytes per dispatch, group counts at 0x200, uniform
/// buffers {address low, high, size} at 0x210 + 16 x slot, storage buffers at 0x310 + 16 x slot.
constexpr u32 kComputeC0Size = 0x500, kComputeC0Groups = 0x200, kComputeC0Uniform = 0x210, kComputeC0Storage = 0x310;
/// The c2 of compute (stage 5) follows the graphics stages' in RenderState::c2.
constexpr u32 kC2Compute = 5;

/// The hardware stage of an NVN graphics stage (vertex 0, tess control 1, tess evaluation 2, geometry 3,
/// fragment 4), which orders c2 and c0's storage-buffer tables.
constexpr u32 HwStage(u32 nvn_stage) {
    constexpr u32 map[5] = {0, 4, 3, 1, 2}; // NVN: vertex, fragment, geometry, tess control, tess evaluation
    return map[nvn_stage];
}
/// The c2 of NVN stage `nvn_stage` in RenderState::c2.
constexpr u32 C2Index(u32 nvn_stage) { return nvn_stage == 5 ? kC2Compute : HwStage(nvn_stage); }

struct RenderTarget {
    TextureState texture;
    ViewArg view;
};

/// c0 as the driver starts it: zero except 0x610 = 0x44fffe00 (2047.94f; its use is not known). 0x650 holds
/// a GPU address inside the driver's own constant memory there, which the layer has no copy of.
inline std::array<u8, kC0Size> InitialC0() {
    std::array<u8, kC0Size> c0{};
    const u32 v = 0x44fffe00;
    std::memcpy(c0.data() + 0x610, &v, 4);
    return c0;
}

/// The c2s as the driver starts them: in each graphics stage's c2 words 0-7 = 0, 1, ... 7 (the driver's
/// start-up table, BRO-SDK 0x58c630 / data 0xb53a70; checked with objdump 2026-10-17, renderer report §6.5;
/// what reads them is not known), the rest zero. Compute's c2 is not known: zero.
inline std::array<std::array<u8, kC2Size>, kHwStages + 1> InitialC2() {
    std::array<std::array<u8, kC2Size>, kHwStages + 1> c2{};
    for (u32 stage = 0; stage < kHwStages; ++stage) {
        for (u32 w = 0; w < 8; ++w) std::memcpy(c2[stage].data() + 4 * w, &w, 4);
    }
    return c2;
}

/// The render state of a queue: NVN keeps it in GPU registers, so it lasts
/// across command lists and submissions until set again. Updated in
/// submission order by the translation (queue.cpp).
struct RenderState {
    u64 device = 0; // the queue's device (clip modes)
    std::array<u8, kC0Size> c0 = InitialC0();
    std::array<std::array<u8, kC2Size>, kHwStages + 1> c2 = InitialC2(); // by hardware stage; [kC2Compute]: compute
    std::array<u32, 3> dispatch{};                             // the group counts of the dispatch being translated
    u32 color_count = 0;
    std::array<RenderTarget, kMaxColorTargets> colors{};
    bool has_depth = false;
    RenderTarget depth{};
    std::array<std::array<float, 4>, kMaxViewports> viewports{}; // x, y, width, height
    std::array<std::array<float, 2>, kMaxViewports> depth_ranges = [] {
        std::array<std::array<float, 2>, kMaxViewports> r;
        r.fill({0.0f, 1.0f}); // NVN's default depth range
        return r;
    }();
    std::array<std::array<s32, 4>, kMaxViewports> scissors{};    // x, y, width, height
    std::array<bool, kMaxViewports> scissor_set{};
    std::array<u8, 24> multisample{}; // the bound NVNmultisampleState, in state.cpp's layout
    bool multisample_set = false;

    // Binds (binds.cpp). State objects as bound, in state.cpp's layouts.
    u32 vertex_attrib_count = 0, vertex_stream_count = 0;
    std::array<u32, 16> vertex_attribs{};               // NVNvertexAttribState (4 bytes each)
    std::array<std::array<u32, 2>, 16> vertex_streams{}; // NVNvertexStreamState: stride, divisor
    std::array<std::array<u64, 2>, 16> vertex_buffers{}; // address, size
    u64 program = 0;                                      // the game's NVNprogram bound for graphics
    std::array<std::optional<ProgramStage>, kStages> stages{}; // its stages as bound (NVN reads them then)
    u64 compute_program = 0;
    std::optional<ProgramStage> compute_stage;
    // Until bound: the objects' SetDefaults values (state.cpp).
    std::array<u8, 8> depth_stencil{0x02, 2, 0x88, 0, 0x11, 0x01, 0x11, 0x01}; // write on, LESS, stencil ALWAYS/KEEP
    std::array<u8, 4> polygon{0, 1, 2, 0};                                    // no culling, CCW, fill
    std::array<u8, 4> color{0, 3, 8, 0};                                      // no blending, COPY, alpha ALWAYS
    std::array<u8, 4> channel_mask{0xFF, 0xFF, 0xFF, 0xFF};                   // every channel written
    std::array<std::array<u8, 8>, kMaxColorTargets> blend = [] {             // ONE / ZERO, ADD; by target
        std::array<std::array<u8, 8>, kMaxColorTargets> b{};
        for (u8 i = 0; i < kMaxColorTargets; ++i) b[i] = {i, 2, 1, 2, 1, 0x11, 0, 0x0C};
        return b;
    }();
    std::array<u32, 2> stencil_ref{}, stencil_mask{}, stencil_value_mask{}; // front, back
    std::array<float, 3> polygon_offset{};                                  // factor, units, clamp
    struct StageBindings {
        std::array<u64, 32> textures{};           // combined handles (BindTexture)
        std::array<u64, 128> separate_textures{};
        std::array<u64, 32> separate_samplers{};
        std::array<u64, 8> images{};
        std::array<std::array<u64, 2>, 18> uniform_buffers{}; // address, size
        std::array<std::array<u64, 2>, 16> storage_buffers{};
    };
    std::array<StageBindings, kStages> bindings{}; // by NVN stage (5: compute)
    u64 texture_pool = 0, sampler_pool = 0;        // the game's pools (SetTexturePool / SetSamplerPool)
    bool depth_bounds_enable = false;
    std::array<float, 2> depth_bounds{0.0f, 1.0f};
};

/// The bound render targets as one framebuffer (render.cpp), for draws.
struct TargetSet {
    VkRenderPass pass;
    VkFramebuffer framebuffer;
    VkExtent2D extent;
    u32 color_count;
    std::array<VkFormat, kMaxColorTargets> colors; // VK_FORMAT_UNDEFINED: no target in the slot
    VkFormat depth;                                // VK_FORMAT_UNDEFINED: none
    u32 scale_n;                                   // the targets' factor x 4 (images.h; 4: the game's size)
};
/// The render pass for these attachments: everything loaded and stored, in VK_IMAGE_LAYOUT_GENERAL (the
/// layer's layout for every image), one subpass; colour slot i is attachment i of the subpass.
VkRenderPass TargetsPass(const std::array<VkFormat, kMaxColorTargets>& colors, u32 color_count, VkFormat depth);
/// The queue's bound render targets: their images (kept alive by `t`), and a framebuffer of TargetsPass.
TargetSet BindTargets(Translator& t);

/// Translates a draw record (draws.cpp); false for any other op.
bool TranslateDrawRecord(Translator& t, Op op, const u8* payload, size_t size);
/// The F9 draw log (window.cpp): draws translated from now on are logged as frame `frame` ("nvn drawlog"
/// lines, draws.cpp); 0 stops it.
void SetDrawLogFrame(u64 frame);
/// The F10 A/B toggle (window.cpp): flips skipping the dispatches of the compute programs listed in
/// RT_NVN_TOGGLE_PROGRAMS (content hashes, as the "new pipeline" lines; comma-separated); returns the new state.
bool FlipToggledPrograms();

/// Applies a bind record (binds.cpp); false for any other op.
bool TranslateBindRecord(Translator& t, Op op, const u8* payload, size_t size);

/// Applies a render-state record (SetRenderTargets, SetViewports,
/// SetDepthRanges, SetScissors, BindMultisampleState) or translates a clear
/// (ClearColor, ClearDepthStencil); false for any other op.
bool TranslateRenderRecord(Translator& t, Op op, const u8* payload, size_t size);

} // namespace rt::nvn
