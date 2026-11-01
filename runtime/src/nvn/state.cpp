// The small state objects: BlendState, ColorState,
// ChannelMaskState, DepthStencilState, MultisampleState, PolygonState,
// VertexAttribState, VertexStreamState. Each keeps the layer's own packed
// fields inline in the game's storage, so the game can copy them by value.
// Sizes are the SDKs' (verified 2026-10-15). Defaults and getter results must
// be the SDK's: RT_NVN_DIFFTEST compares every function here with the game's
// own. Enum values are NVN's and fit the fields below (all < 16 unless noted).
#include "state.h"

namespace rt::nvn {

namespace {

template <typename T, typename F>
void Modify(u64 guest, F f) {
    T v = Load<T>(guest);
    f(v);
    Store(guest, v);
}

constexpr u32 kFront = 1, kBack = 2; // NVNface bits

} // namespace

// --- BlendState -------------------------------------------------------------------------------

NVN_PROC(nvnBlendStateSetDefaults, void, u64 s) {
    // ONE / ZERO for color and alpha, ADD, no advanced mode
    Store(s, BlendState{0, 2, 1, 2, 1, 0x11, 0, 0x0C});
}
NVN_PROC(nvnBlendStateSetBlendTarget, void, u64 s, u32 target) {
    Modify<BlendState>(s, [&](BlendState& b) { b.target = static_cast<u8>(target); });
}
NVN_PROC(nvnBlendStateSetBlendFunc, void, u64 s, u32 src, u32 dst, u32 src_alpha, u32 dst_alpha) {
    Modify<BlendState>(s, [&](BlendState& b) {
        b.src = static_cast<u8>(src);
        b.dst = static_cast<u8>(dst);
        b.src_alpha = static_cast<u8>(src_alpha);
        b.dst_alpha = static_cast<u8>(dst_alpha);
    });
}
NVN_PROC(nvnBlendStateSetBlendEquation, void, u64 s, u32 color, u32 alpha) {
    Modify<BlendState>(s, [&](BlendState& b) { b.equations = static_cast<u8>((color & 15) | (alpha & 15) << 4); });
}
NVN_PROC(nvnBlendStateSetAdvancedMode, void, u64 s, u32 mode) {
    Modify<BlendState>(s, [&](BlendState& b) { b.advanced_mode = static_cast<u8>(mode); });
}
NVN_PROC(nvnBlendStateSetAdvancedOverlap, void, u64 s, u32 overlap) {
    Modify<BlendState>(s, [&](BlendState& b) { b.advanced = static_cast<u8>((b.advanced & ~3) | (overlap & 3)); });
}
NVN_PROC(nvnBlendStateSetAdvancedPremultipliedSrc, void, u64 s, bool on) {
    Modify<BlendState>(s, [&](BlendState& b) { b.advanced = static_cast<u8>((b.advanced & ~4) | (on ? 4 : 0)); });
}
NVN_PROC(nvnBlendStateSetAdvancedNormalizedDst, void, u64 s, bool on) {
    Modify<BlendState>(s, [&](BlendState& b) { b.advanced = static_cast<u8>((b.advanced & ~8) | (on ? 8 : 0)); });
}
NVN_PROC(nvnBlendStateGetBlendTarget, s32, u64 s) { return Load<BlendState>(s).target; }
NVN_PROC(nvnBlendStateGetBlendFunc, void, u64 s, u64 src, u64 dst, u64 src_alpha, u64 dst_alpha) {
    const auto b = Load<BlendState>(s);
    hle::WriteGuest<u32>(src, b.src);
    hle::WriteGuest<u32>(dst, b.dst);
    hle::WriteGuest<u32>(src_alpha, b.src_alpha);
    hle::WriteGuest<u32>(dst_alpha, b.dst_alpha);
}
NVN_PROC(nvnBlendStateGetBlendEquation, void, u64 s, u64 color, u64 alpha) {
    const auto b = Load<BlendState>(s);
    hle::WriteGuest<u32>(color, b.equations & 15);
    hle::WriteGuest<u32>(alpha, b.equations >> 4);
}
NVN_PROC(nvnBlendStateGetAdvancedMode, u32, u64 s) { return Load<BlendState>(s).advanced_mode; }
NVN_PROC(nvnBlendStateGetAdvancedOverlap, u32, u64 s) { return Load<BlendState>(s).advanced & 3; }
NVN_PROC(nvnBlendStateGetAdvancedPremultipliedSrc, bool, u64 s) { return Load<BlendState>(s).advanced & 4; }
NVN_PROC(nvnBlendStateGetAdvancedNormalizedDst, bool, u64 s) { return Load<BlendState>(s).advanced & 8; }

// --- ColorState -------------------------------------------------------------------------------

NVN_PROC(nvnColorStateSetDefaults, void, u64 s) { Store(s, ColorState{0, 3, 8, 0}); } // COPY, ALWAYS
NVN_PROC(nvnColorStateSetBlendEnable, void, u64 s, s32 index, bool on) {
    Modify<ColorState>(s, [&](ColorState& v) {
        const u8 bit = static_cast<u8>(1u << (index & 7));
        v.blend_enable = static_cast<u8>(on ? v.blend_enable | bit : v.blend_enable & ~bit);
    });
}
NVN_PROC(nvnColorStateSetLogicOp, void, u64 s, u32 op) {
    Modify<ColorState>(s, [&](ColorState& v) { v.logic_op = static_cast<u8>(op); });
}
NVN_PROC(nvnColorStateSetAlphaTest, void, u64 s, u32 func) {
    Modify<ColorState>(s, [&](ColorState& v) { v.alpha_test = static_cast<u8>(func); });
}
NVN_PROC(nvnColorStateGetBlendEnable, bool, u64 s, s32 index) {
    return (Load<ColorState>(s).blend_enable >> (index & 7)) & 1;
}
NVN_PROC(nvnColorStateGetLogicOp, u32, u64 s) { return Load<ColorState>(s).logic_op; }
NVN_PROC(nvnColorStateGetAlphaTest, u32, u64 s) { return Load<ColorState>(s).alpha_test; }

// --- ChannelMaskState ---------------------------------------------------------------------------

NVN_PROC(nvnChannelMaskStateSetDefaults, void, u64 s) { Store(s, ChannelMaskState{0xFFFFFFFF}); }
NVN_PROC(nvnChannelMaskStateSetChannelMask, void, u64 s, s32 index, bool r, bool g, bool b, bool a) {
    Modify<ChannelMaskState>(s, [&](ChannelMaskState& v) {
        const u32 shift = 4 * static_cast<u32>(index & 7);
        const u32 m = (r ? 1 : 0) | (g ? 2 : 0) | (b ? 4 : 0) | (a ? 8 : 0);
        v.masks = (v.masks & ~(15u << shift)) | m << shift;
    });
}
NVN_PROC(nvnChannelMaskStateGetChannelMask, void, u64 s, s32 index, u64 r, u64 g, u64 b, u64 a) {
    const u32 m = Load<ChannelMaskState>(s).masks >> (4 * static_cast<u32>(index & 7));
    hle::WriteGuest<u8>(r, m & 1);
    hle::WriteGuest<u8>(g, (m >> 1) & 1);
    hle::WriteGuest<u8>(b, (m >> 2) & 1);
    hle::WriteGuest<u8>(a, (m >> 3) & 1);
}

// --- DepthStencilState --------------------------------------------------------------------------

NVN_PROC(nvnDepthStencilStateSetDefaults, void, u64 s) {
    // depth test off, depth write on, LESS; stencil off, ALWAYS, KEEP
    Store(s, DepthStencilState{0x02, 2, 0x88, 0, 0x111, 0x111});
}
NVN_PROC(nvnDepthStencilStateSetDepthTestEnable, void, u64 s, bool on) {
    Modify<DepthStencilState>(s, [&](DepthStencilState& v) { v.flags = static_cast<u8>((v.flags & ~1) | (on ? 1 : 0)); });
}
NVN_PROC(nvnDepthStencilStateSetDepthWriteEnable, void, u64 s, bool on) {
    Modify<DepthStencilState>(s, [&](DepthStencilState& v) { v.flags = static_cast<u8>((v.flags & ~2) | (on ? 2 : 0)); });
}
NVN_PROC(nvnDepthStencilStateSetStencilTestEnable, void, u64 s, bool on) {
    Modify<DepthStencilState>(s, [&](DepthStencilState& v) { v.flags = static_cast<u8>((v.flags & ~4) | (on ? 4 : 0)); });
}
NVN_PROC(nvnDepthStencilStateSetDepthFunc, void, u64 s, u32 func) {
    Modify<DepthStencilState>(s, [&](DepthStencilState& v) { v.depth_func = static_cast<u8>(func); });
}
NVN_PROC(nvnDepthStencilStateSetStencilFunc, void, u64 s, u32 face, u32 func) {
    Modify<DepthStencilState>(s, [&](DepthStencilState& v) {
        if (face & kFront) v.stencil_funcs = static_cast<u8>((v.stencil_funcs & 0xF0) | (func & 15));
        if (face & kBack) v.stencil_funcs = static_cast<u8>((v.stencil_funcs & 0x0F) | (func & 15) << 4);
    });
}
NVN_PROC(nvnDepthStencilStateSetStencilOp, void, u64 s, u32 face, u32 fail, u32 depth_fail, u32 depth_pass) {
    Modify<DepthStencilState>(s, [&](DepthStencilState& v) {
        const u16 ops = static_cast<u16>((fail & 15) | (depth_fail & 15) << 4 | (depth_pass & 15) << 8);
        if (face & kFront) v.front_ops = ops;
        if (face & kBack) v.back_ops = ops;
    });
}
NVN_PROC(nvnDepthStencilStateGetDepthTestEnable, bool, u64 s) { return Load<DepthStencilState>(s).flags & 1; }
NVN_PROC(nvnDepthStencilStateGetDepthWriteEnable, bool, u64 s) { return Load<DepthStencilState>(s).flags & 2; }
NVN_PROC(nvnDepthStencilStateGetStencilTestEnable, bool, u64 s) { return Load<DepthStencilState>(s).flags & 4; }
NVN_PROC(nvnDepthStencilStateGetDepthFunc, u32, u64 s) { return Load<DepthStencilState>(s).depth_func; }
NVN_PROC(nvnDepthStencilStateGetStencilFunc, u32, u64 s, u32 face) {
    const auto v = Load<DepthStencilState>(s);
    return face == kBack ? v.stencil_funcs >> 4 : v.stencil_funcs & 15;
}
NVN_PROC(nvnDepthStencilStateGetStencilOp, void, u64 s, u32 face, u64 fail, u64 depth_fail, u64 depth_pass) {
    const auto v = Load<DepthStencilState>(s);
    const u16 ops = face == kBack ? v.back_ops : v.front_ops;
    hle::WriteGuest<u32>(fail, ops & 15);
    hle::WriteGuest<u32>(depth_fail, (ops >> 4) & 15);
    hle::WriteGuest<u32>(depth_pass, (ops >> 8) & 15);
}

// --- MultisampleState ---------------------------------------------------------------------------

NVN_PROC(nvnMultisampleStateSetDefaults, void, u64 s) {
    MultisampleState v{};
    v.flags = 0x05; // multisample on, alpha-to-coverage dither on
    Store(s, v);
}
#define MS_FLAG(set, get, bit)                                                                                  \
    NVN_PROC(set, void, u64 s, bool on) {                                                                       \
        Modify<MultisampleState>(s, [&](MultisampleState& v) {                                                  \
            v.flags = static_cast<u8>((v.flags & ~(1u << (bit))) | (on ? 1u << (bit) : 0));                     \
        });                                                                                                     \
    }                                                                                                           \
    NVN_PROC(get, bool, u64 s) { return (Load<MultisampleState>(s).flags >> (bit)) & 1; }
MS_FLAG(nvnMultisampleStateSetMultisampleEnable, nvnMultisampleStateGetMultisampleEnable, 0)
MS_FLAG(nvnMultisampleStateSetAlphaToCoverageEnable, nvnMultisampleStateGetAlphaToCoverageEnable, 1)
MS_FLAG(nvnMultisampleStateSetAlphaToCoverageDither, nvnMultisampleStateGetAlphaToCoverageDither, 2)
MS_FLAG(nvnMultisampleStateSetCoverageToColorEnable, nvnMultisampleStateGetCoverageToColorEnable, 3)
MS_FLAG(nvnMultisampleStateSetSampleLocationsEnable, nvnMultisampleStateGetSampleLocationsEnable, 4)
MS_FLAG(nvnMultisampleStateSetSampleLocationsGridEnable, nvnMultisampleStateGetSampleLocationsGridEnable, 5)
#undef MS_FLAG
#define MS_FIELD(set, get, field)                                                                               \
    NVN_PROC(set, void, u64 s, s32 n) {                                                                         \
        Modify<MultisampleState>(s, [&](MultisampleState& v) { v.field = static_cast<u8>(n); });                \
    }                                                                                                           \
    NVN_PROC(get, s32, u64 s) { return Load<MultisampleState>(s).field; }
MS_FIELD(nvnMultisampleStateSetSamples, nvnMultisampleStateGetSamples, samples)
MS_FIELD(nvnMultisampleStateSetRasterSamples, nvnMultisampleStateGetRasterSamples, raster_samples)
MS_FIELD(nvnMultisampleStateSetCoverageToColorOutput, nvnMultisampleStateGetCoverageToColorOutput,
         coverage_to_color_output)
MS_FIELD(nvnMultisampleStateSetCoverageModulationMode, nvnMultisampleStateGetCoverageModulationMode,
         coverage_modulation)
#undef MS_FIELD

// --- PolygonState -------------------------------------------------------------------------------

NVN_PROC(nvnPolygonStateSetDefaults, void, u64 s) { Store(s, PolygonState{0, 1, 2, 0}); } // NONE, CCW, FILL
NVN_PROC(nvnPolygonStateSetCullFace, void, u64 s, u32 v) {
    Modify<PolygonState>(s, [&](PolygonState& p) { p.cull_face = static_cast<u8>(v); });
}
NVN_PROC(nvnPolygonStateSetFrontFace, void, u64 s, u32 v) {
    Modify<PolygonState>(s, [&](PolygonState& p) { p.front_face = static_cast<u8>(v); });
}
NVN_PROC(nvnPolygonStateSetPolygonMode, void, u64 s, u32 v) {
    Modify<PolygonState>(s, [&](PolygonState& p) { p.polygon_mode = static_cast<u8>(v); });
}
NVN_PROC(nvnPolygonStateSetPolygonOffsetEnables, void, u64 s, u32 v) {
    Modify<PolygonState>(s, [&](PolygonState& p) { p.offset_enables = static_cast<u8>(v); });
}
NVN_PROC(nvnPolygonStateGetCullFace, u32, u64 s) { return Load<PolygonState>(s).cull_face; }
NVN_PROC(nvnPolygonStateGetFrontFace, u32, u64 s) { return Load<PolygonState>(s).front_face; }
NVN_PROC(nvnPolygonStateGetPolygonMode, u32, u64 s) { return Load<PolygonState>(s).polygon_mode; }
NVN_PROC(nvnPolygonStateGetPolygonOffsetEnables, u32, u64 s) { return Load<PolygonState>(s).offset_enables; }

// --- VertexAttribState --------------------------------------------------------------------------

NVN_PROC(nvnVertexAttribStateSetDefaults, void, u64 s) { Store(s, VertexAttribState{0}); }
NVN_PROC(nvnVertexAttribStateSetFormat, void, u64 s, u32 format, s64 offset) {
    Modify<VertexAttribState>(s, [&](VertexAttribState& v) {
        v.bits = (v.bits & 0x1F00) | (format & 0xFF) | static_cast<u32>(offset) << 13;
    });
}
NVN_PROC(nvnVertexAttribStateSetStreamIndex, void, u64 s, s32 index) {
    Modify<VertexAttribState>(s, [&](VertexAttribState& v) {
        v.bits = (v.bits & ~0x1F00u) | (static_cast<u32>(index) & 0x1F) << 8;
    });
}
NVN_PROC(nvnVertexAttribStateGetFormat, void, u64 s, u64 format, u64 offset) {
    const u32 bits = Load<VertexAttribState>(s).bits;
    hle::WriteGuest<u32>(format, bits & 0xFF);
    hle::WriteGuest<s64>(offset, bits >> 13);
}
NVN_PROC(nvnVertexAttribStateGetStreamIndex, s32, u64 s) { return (Load<VertexAttribState>(s).bits >> 8) & 0x1F; }

// --- VertexStreamState --------------------------------------------------------------------------

NVN_PROC(nvnVertexStreamStateSetDefaults, void, u64 s) { Store(s, VertexStreamState{0, 0}); }
NVN_PROC(nvnVertexStreamStateSetStride, void, u64 s, s64 stride) {
    Modify<VertexStreamState>(s, [&](VertexStreamState& v) { v.stride = static_cast<u32>(stride); });
}
NVN_PROC(nvnVertexStreamStateSetDivisor, void, u64 s, s32 divisor) {
    Modify<VertexStreamState>(s, [&](VertexStreamState& v) { v.divisor = static_cast<u32>(divisor); });
}
NVN_PROC(nvnVertexStreamStateGetStride, s64, u64 s) { return Load<VertexStreamState>(s).stride; }
NVN_PROC(nvnVertexStreamStateGetDivisor, s32, u64 s) { return static_cast<s32>(Load<VertexStreamState>(s).divisor); }

} // namespace rt::nvn
