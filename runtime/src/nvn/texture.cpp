// NVNtexturePool: so far the pool itself. Textures, their
// registration and views follow in Phase 6; until then those calls stop the run.
//
// A texture pool is a host table of texture entries by id; NVN also writes
// hardware descriptors into the pool's memory, which only the GPU reads, so
// the layer leaves that memory alone.
#include "texture.h"

#include "../threads.h"
#include "tap.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace rt::nvn {

namespace {

/// A registered pool entry: copies of the texture and view as they were then
/// (NVN writes a descriptor from them at registration).
struct PoolEntry {
    TextureState texture;
    ViewArg view;
    bool image;
    bool registered;
};

struct TexturePool : Object {
    static constexpr Type kType = Type::kTexturePool;
    u64 memory_pool = 0;
    s64 offset = 0;
    s32 size = 0;
    std::vector<PoolEntry> entries;
};

constexpr u64 kDescriptorSize = 32; // NVN's, per entry of the pool's memory

} // namespace

NVN_PROC(nvnTexturePoolInitialize, bool, u64 pool, u64 memory_pool, s64 offset, s32 count) {
    const PoolRange p = PoolOf(memory_pool);
    if (offset < 0 || count < 0 || static_cast<u64>(offset) + kDescriptorSize * static_cast<u64>(count) > p.size) {
        Fatal("nvn: nvnTexturePoolInitialize: %d descriptors at 0x%llx do not fit the memory pool (0x%llx bytes)",
              count, (unsigned long long)offset, (unsigned long long)p.size);
    }
    TexturePool* o = Create<TexturePool>(pool);
    o->memory_pool = memory_pool;
    o->offset = offset;
    o->size = count;
    o->entries.resize(static_cast<size_t>(count));
    return true;
}
NVN_PROC(nvnTexturePoolFinalize, void, u64 pool) { Destroy<TexturePool>(pool); }
NVN_PROC(nvnTexturePoolSetDebugLabel, void, u64 pool, u64 label) { Get<TexturePool>(pool); }
NVN_PROC(nvnTexturePoolGetMemoryPool, u64, u64 pool) { return Get<TexturePool>(pool)->memory_pool; }
NVN_PROC(nvnTexturePoolGetMemoryOffset, s64, u64 pool) { return Get<TexturePool>(pool)->offset; }
NVN_PROC(nvnTexturePoolGetSize, s32, u64 pool) { return Get<TexturePool>(pool)->size; }

// --- TextureBuilder ---------------------------------------------------------------------------
// Inline in the game's NVNtextureBuilder (texture.h). Defaults as the SDK's (RT_NVN_DIFFTEST).

namespace {

template <typename F>
void ChangeBuilder(u64 builder, F&& f) {
    auto b = Load<TextureBuilder>(builder);
    f(b);
    Store(builder, b);
}

u8 SwizzleField(u32 v) {
    if (v > 0xff) Fatal("nvn: texture swizzle %u is no NVN value", v);
    return static_cast<u8>(v);
}

} // namespace

NVN_PROC(nvnTextureBuilderSetDevice, void, u64 builder, u64 device) {
    ChangeBuilder(builder, [&](TextureBuilder& b) { b.device = device; });
}
NVN_PROC(nvnTextureBuilderSetDefaults, void, u64 builder) {
    ChangeBuilder(builder, [](TextureBuilder& b) {
        const u64 device = b.device;
        b = TextureBuilder{};
        b.device = device;
        b.target = 1; // NVN_TEXTURE_TARGET_2D
        b.levels = 1;
        b.swizzle[0] = 2, b.swizzle[1] = 3, b.swizzle[2] = 4, b.swizzle[3] = 5; // R, G, B, A
    });
}
#define TB_SET(fn, field)                                                                                           \
    NVN_PROC(fn, void, u64 builder, u32 v) {                                                                        \
        ChangeBuilder(builder, [&](TextureBuilder& b) { b.field = v; });                                            \
    }
TB_SET(nvnTextureBuilderSetFlags, flags)
TB_SET(nvnTextureBuilderSetTarget, target)
TB_SET(nvnTextureBuilderSetWidth, width)
TB_SET(nvnTextureBuilderSetHeight, height)
TB_SET(nvnTextureBuilderSetDepth, depth)
TB_SET(nvnTextureBuilderSetLevels, levels)
TB_SET(nvnTextureBuilderSetFormat, format)
TB_SET(nvnTextureBuilderSetSamples, samples)
TB_SET(nvnTextureBuilderSetDepthStencilMode, ds_mode)
TB_SET(nvnTextureBuilderSetGLTextureName, gl_name)
#undef TB_SET
NVN_PROC(nvnTextureBuilderSetSize1D, void, u64 builder, u32 w) {
    ChangeBuilder(builder, [&](TextureBuilder& b) { b.width = w, b.height = 1, b.depth = 1; });
}
NVN_PROC(nvnTextureBuilderSetSize2D, void, u64 builder, u32 w, u32 h) {
    ChangeBuilder(builder, [&](TextureBuilder& b) { b.width = w, b.height = h, b.depth = 1; });
}
NVN_PROC(nvnTextureBuilderSetSize3D, void, u64 builder, u32 w, u32 h, u32 d) {
    ChangeBuilder(builder, [&](TextureBuilder& b) { b.width = w, b.height = h, b.depth = d; });
}
NVN_PROC(nvnTextureBuilderSetStride, void, u64 builder, s64 stride) {
    ChangeBuilder(builder, [&](TextureBuilder& b) { b.stride = static_cast<s32>(stride); });
}
NVN_PROC(nvnTextureBuilderSetStorage, void, u64 builder, u64 pool, s64 offset) {
    ChangeBuilder(builder, [&](TextureBuilder& b) { b.pool = pool, b.offset = offset; });
}
NVN_PROC(nvnTextureBuilderSetPackagedTextureData, void, u64 builder, u64 data) {
    ChangeBuilder(builder, [&](TextureBuilder& b) { b.packaged_data = data; });
}
NVN_PROC(nvnTextureBuilderSetPackagedTextureLayout, void, u64 builder, u64 layout) {
    const u64 v = layout ? hle::ReadGuest<u32>(layout) & 0xfff : 0;
    ChangeBuilder(builder, [&](TextureBuilder& b) { b.layout = v, b.layout_set = layout != 0; });
}
NVN_PROC(nvnTextureBuilderSetSwizzle, void, u64 builder, u32 r, u32 g, u32 b_, u32 a) {
    ChangeBuilder(builder, [&](TextureBuilder& b) {
        b.swizzle[0] = SwizzleField(r), b.swizzle[1] = SwizzleField(g);
        b.swizzle[2] = SwizzleField(b_), b.swizzle[3] = SwizzleField(a);
    });
}

#define TB_GET(fn, type, field)                                                                                     \
    NVN_PROC(fn, type, u64 builder) { return static_cast<type>(Load<TextureBuilder>(builder).field); }
TB_GET(nvnTextureBuilderGetDevice, u64, device)
TB_GET(nvnTextureBuilderGetFlags, u32, flags)
TB_GET(nvnTextureBuilderGetTarget, u32, target)
TB_GET(nvnTextureBuilderGetWidth, u32, width)
TB_GET(nvnTextureBuilderGetHeight, u32, height)
TB_GET(nvnTextureBuilderGetDepth, u32, depth)
TB_GET(nvnTextureBuilderGetLevels, u32, levels)
TB_GET(nvnTextureBuilderGetFormat, u32, format)
TB_GET(nvnTextureBuilderGetSamples, u32, samples)
TB_GET(nvnTextureBuilderGetDepthStencilMode, u32, ds_mode)
TB_GET(nvnTextureBuilderGetStride, s64, stride)
TB_GET(nvnTextureBuilderGetGLTextureName, u32, gl_name)
TB_GET(nvnTextureBuilderGetMemoryPool, u64, pool)
TB_GET(nvnTextureBuilderGetMemoryOffset, s64, offset)
TB_GET(nvnTextureBuilderGetPackagedTextureData, u64, packaged_data)
#undef TB_GET
NVN_PROC(nvnTextureBuilderGetSwizzle, void, u64 builder, u64 r, u64 g, u64 b, u64 a) {
    const auto tb = Load<TextureBuilder>(builder);
    hle::WriteGuest<u32>(r, tb.swizzle[0]);
    hle::WriteGuest<u32>(g, tb.swizzle[1]);
    hle::WriteGuest<u32>(b, tb.swizzle[2]);
    hle::WriteGuest<u32>(a, tb.swizzle[3]);
}
// Writes the layout word as NVN keeps it (its low 12 bits; 0 if none) and
// answers 1 if one was set.
NVN_PROC(nvnTextureBuilderGetPackagedTextureLayout, u32, u64 builder, u64 out) {
    const auto tb = Load<TextureBuilder>(builder);
    hle::WriteGuest<u32>(out, static_cast<u32>(tb.layout_set ? tb.layout : 0));
    return tb.layout_set;
}

// --- Storage size and alignment -------------------------------------------
// NVN's rules, checked against the real driver's answers for every texture of
// the title route (1,088 of 1,088, tap mode):
// - Block-linear: GOBs of 64 bytes x 8 rows. Block height (log2 GOBs) from
//   1.5 x the level-0 height in format blocks: < 16 -> 0, < 32 -> 1, < 64 -> 2,
//   < 128 -> 3, else 4 (SDK 0x5a5ef0); 3D textures: block height 0 and block
//   depth from 1.5 x depth (< 2, < 4, < 8, < 16, else 4). Per level both are
//   lowered while the level fits in half a block. Levels follow each other;
//   arrays and cube maps round each layer to the level-0 block size.
// - Sparse (flag 0x20): a level at least one 64 KB tile wide and tall (in
//   blocks: 256x256 for 1-byte blocks, 256x128, 128x128, 128x64, 64x64 for 16)
//   takes whole tiles; from the first smaller level on, block-linear.
// - Alignment: 128 linear render target (0x40), 32 linear (0x10), else 64 KB
//   for compressible memory (flag 0x08 and a depth/stencil format or an
//   uncompressed one of >= 4 bytes; the Switch's page size for it), else 512.
//   The size is rounded up to the alignment.
// - Linear (flags 0x10, 0x40): stride x height; buffer textures: width x bytes
//   per texel (rules read from the SDK).
// Checked in tap mode on the title route: every builder size/alignment query
// and every texture's size after nvnTextureInitialize (linear and buffer
// textures included), 0 differences. Unchecked (not on the route): 1D and
// cube-map arrays (logged once); multisample and packaged layouts stop the run.

namespace {

constexpr u32 kTarget3D = 2, kTarget2DArray = 4, kTargetCube = 8, kTargetCubeArray = 9, kTargetBuffer = 10;
constexpr u32 kFlagCompressible = 0x08, kFlagLinear = 0x10, kFlagSparse = 0x20, kFlagLinearRT = 0x40;

u64 CeilDiv(u64 a, u64 b) { return (a + b - 1) / b; }

u32 BlockHeightLog2(u64 dim) {
    const u64 v = dim + dim / 2;
    return v < 16 ? 0 : v < 32 ? 1 : v < 64 ? 2 : v < 128 ? 3 : 4;
}
u32 BlockDepthLog2(u64 dim) {
    const u64 v = dim + dim / 2;
    return v < 2 ? 0 : v < 4 ? 1 : v < 8 ? 2 : v < 16 ? 3 : 4;
}

void Unchecked(const char* what) {
    static std::mutex mu;
    static std::map<std::string, bool> told;
    std::lock_guard lock(mu);
    if (!told[what]) {
        told[what] = true;
        Log("nvn: texture storage of %s computed by a rule no game has checked yet (RT_NVN_TAP compares it)", what);
    }
}

} // namespace

std::optional<StorageLayout> ComputeLayout(const TextureBuilder& b, std::string& why) {
    char buf[96];
    const FormatInfo* f = FindFormat(b.format);
    if (!f) {
        std::snprintf(buf, sizeof buf, "texture format 0x%x is not known to the layer yet", b.format);
        return why = buf, std::nullopt;
    }
    if (b.samples > 1) {
        std::snprintf(buf, sizeof buf, "multisample texture storage is not implemented yet (%u samples)", b.samples);
        return why = buf, std::nullopt;
    }
    if (b.layout_set) return why = "packaged texture layouts are not implemented yet", std::nullopt;
    if (b.flags & (kFlagLinear | kFlagLinearRT)) {
        const u32 alignment = b.flags & kFlagLinearRT ? 128 : 32;
        const u64 size = static_cast<u64>(static_cast<u32>(b.stride)) * b.height;
        return StorageLayout{{CeilDiv(size, alignment) * alignment, alignment}, {0, size}, size};
    }
    if (b.target == kTargetBuffer) {
        const u64 size = static_cast<u64>(b.width) * f->bytes;
        return StorageLayout{{size, f->bytes == 12 ? 4u : f->bytes}, {0, size}, size};
    }
    const bool compressible = (b.flags & kFlagCompressible) && (f->depth_stencil || (f->block_w == 1 && f->bytes >= 4));
    const u32 alignment = compressible ? 0x10000 : 512;
    if (b.target == kTargetCubeArray) Unchecked("cube map arrays");
    if (b.target == 3) Unchecked("1D arrays");
    const u64 layers = ArrayLayers(b);
    const u64 height = b.target == 3 ? 1 : b.height;
    const u64 depth = b.target == kTarget3D ? b.depth : 1;
    // 3D: block height 0, except sparse 3D textures: the height solver's, at most 2 (fitted 2026-10-16 on 17 3D
    // sizes and the 4 slice offsets of the title route, 4- and 8-byte formats; the cap may depend on the format)
    const u32 bh0 = b.target != kTarget3D       ? BlockHeightLog2(CeilDiv(height, f->block_h))
                    : (b.flags & kFlagSparse) ? std::min(BlockHeightLog2(CeilDiv(height, f->block_h)), 2u)
                                              : 0;
    const u32 bd0 = b.target == kTarget3D ? BlockDepthLog2(depth) : 0;
    u64 tile_w = 0, tile_h = 0; // the 64 KB sparse tile in blocks
    switch (f->bytes) {
    case 1: tile_w = 256, tile_h = 256; break;
    case 2: tile_w = 256, tile_h = 128; break;
    case 4: tile_w = 128, tile_h = 128; break;
    case 8: tile_w = 128, tile_h = 64; break;
    case 16: tile_w = 64, tile_h = 64; break;
    }
    if ((b.flags & kFlagSparse) && !tile_w) return why = "a sparse texture with 3-, 6- or 12-byte blocks", std::nullopt;
    u64 total = 0;
    std::vector<u64> offsets;
    bool tail = !(b.flags & kFlagSparse);
    for (u32 l = 0; l < std::max(b.levels, 1u); ++l) {
        offsets.push_back(total);
        const u64 wl = std::max<u64>(1, b.width >> l), hl = std::max<u64>(1, height >> l),
                  dl = std::max<u64>(1, depth >> l);
        const u64 wb = CeilDiv(wl, f->block_w), hb = CeilDiv(hl, f->block_h);
        if (!tail) {
            if (wb >= tile_w && hb >= tile_h) {
                total += CeilDiv(wb, tile_w) * CeilDiv(hb, tile_h) * dl * 0x10000;
                continue;
            }
            tail = true;
        }
        const u64 hg = CeilDiv(hb, 8);
        u32 bh = bh0, bd = bd0;
        while (bh > 0 && hg <= (1ULL << (bh - 1))) --bh;
        while (bd > 0 && dl <= (1ULL << (bd - 1))) --bd;
        total += CeilDiv(wb * f->bytes, 64) * CeilDiv(hg, 1ULL << bh) * CeilDiv(dl, 1ULL << bd) * (512ULL << (bh + bd));
    }
    offsets.push_back(total); // the end of the levels (the game asks for "level = levels")
    u64 layer_size = total;
    if (layers > 1) {
        const u64 block0 = 512ULL << (bh0 + bd0);
        layer_size = CeilDiv(total, block0) * block0;
        total = layer_size * layers;
    }
    return StorageLayout{{CeilDiv(total, alignment) * alignment, alignment}, std::move(offsets), layer_size};
}

std::optional<Storage> ComputeStorage(const TextureBuilder& b, std::string& why) {
    const auto l = ComputeLayout(b, why);
    if (!l) return std::nullopt;
    return l->storage;
}

namespace {

Storage StorageOrStop(u64 builder) {
    std::string why;
    const auto s = ComputeStorage(Load<TextureBuilder>(builder), why);
    if (!s) Fatal("nvn: %s", why.c_str());
    return *s;
}

} // namespace

NVN_PROC(nvnTextureBuilderGetStorageSize, u64, u64 builder) { return StorageOrStop(builder).size; }
NVN_PROC(nvnTextureBuilderGetStorageAlignment, u32, u64 builder) { return StorageOrStop(builder).alignment; }

// --- TextureView -----------------------------------------------------------------
// Inline in the game's NVNtextureView (texture.h); a property counts only once
// set, and each getter answers whether it was (as the SDK's).

namespace {

template <typename F>
void ChangeView(u64 view, F&& f) {
    auto v = Load<TextureView>(view);
    f(v);
    Store(view, v);
}

} // namespace

NVN_PROC(nvnTextureViewSetDefaults, void, u64 view) { Store(view, TextureView{}); }
NVN_PROC(nvnTextureViewSetLevels, void, u64 view, s32 base, s32 count) {
    ChangeView(view, [&](TextureView& v) {
        v.base_level = static_cast<u8>(base), v.num_levels = static_cast<u8>(count), v.set |= kViewLevels;
    });
}
NVN_PROC(nvnTextureViewSetLayers, void, u64 view, s32 base, s32 count) {
    ChangeView(view, [&](TextureView& v) {
        v.base_layer = static_cast<int16_t>(base), v.num_layers = static_cast<int16_t>(count), v.set |= kViewLayers;
    });
}
NVN_PROC(nvnTextureViewSetFormat, void, u64 view, u32 format) {
    ChangeView(view, [&](TextureView& v) { v.format = format, v.set |= kViewFormat; });
}
NVN_PROC(nvnTextureViewSetSwizzle, void, u64 view, u32 r, u32 g, u32 b, u32 a) {
    ChangeView(view, [&](TextureView& v) {
        v.swizzle[0] = static_cast<u8>(r), v.swizzle[1] = static_cast<u8>(g);
        v.swizzle[2] = static_cast<u8>(b), v.swizzle[3] = static_cast<u8>(a);
        v.set |= kViewSwizzle;
    });
}
NVN_PROC(nvnTextureViewSetDepthStencilMode, void, u64 view, u32 mode) {
    ChangeView(view, [&](TextureView& v) { v.set = static_cast<u8>((v.set & 0x3f) | kViewDsMode | (mode << 6)); });
}
NVN_PROC(nvnTextureViewSetTarget, void, u64 view, u32 target) {
    ChangeView(view, [&](TextureView& v) { v.target = static_cast<u8>(target), v.set |= kViewTarget; });
}
NVN_PROC(nvnTextureViewGetLevels, bool, u64 view, u64 base, u64 count) {
    const auto v = Load<TextureView>(view);
    hle::WriteGuest<s32>(base, v.base_level);
    hle::WriteGuest<s32>(count, v.num_levels);
    return v.set & kViewLevels;
}
NVN_PROC(nvnTextureViewGetLayers, bool, u64 view, u64 base, u64 count) {
    const auto v = Load<TextureView>(view);
    hle::WriteGuest<s32>(base, v.base_layer);
    hle::WriteGuest<s32>(count, v.num_layers);
    return v.set & kViewLayers;
}
NVN_PROC(nvnTextureViewGetFormat, bool, u64 view, u64 format) {
    const auto v = Load<TextureView>(view);
    hle::WriteGuest<u32>(format, v.format);
    return v.set & kViewFormat;
}
NVN_PROC(nvnTextureViewGetSwizzle, bool, u64 view, u64 r, u64 g, u64 b, u64 a) {
    const auto v = Load<TextureView>(view);
    hle::WriteGuest<u32>(r, v.swizzle[0]);
    hle::WriteGuest<u32>(g, v.swizzle[1]);
    hle::WriteGuest<u32>(b, v.swizzle[2]);
    hle::WriteGuest<u32>(a, v.swizzle[3]);
    return v.set & kViewSwizzle;
}
NVN_PROC(nvnTextureViewGetDepthStencilMode, bool, u64 view, u64 mode) {
    const auto v = Load<TextureView>(view);
    hle::WriteGuest<u32>(mode, v.set >> 6);
    return v.set & kViewDsMode;
}
NVN_PROC(nvnTextureViewGetTarget, bool, u64 view, u64 target) {
    const auto v = Load<TextureView>(view);
    hle::WriteGuest<u32>(target, v.target);
    return v.set & kViewTarget;
}
// Equal when the same properties are set and each set property is equal.
NVN_PROC(nvnTextureViewCompare, bool, u64 a, u64 b) {
    const auto x = Load<TextureView>(a), y = Load<TextureView>(b);
    if ((x.set & 0x3f) != (y.set & 0x3f)) return false;
    if ((x.set & kViewLevels) && (x.base_level != y.base_level || x.num_levels != y.num_levels)) return false;
    if ((x.set & kViewLayers) && (x.base_layer != y.base_layer || x.num_layers != y.num_layers)) return false;
    if ((x.set & kViewFormat) && x.format != y.format) return false;
    if ((x.set & kViewSwizzle) && std::memcmp(x.swizzle, y.swizzle, 4) != 0) return false;
    if ((x.set & kViewDsMode) && (x.set >> 6) != (y.set >> 6)) return false;
    if ((x.set & kViewTarget) && x.target != y.target) return false;
    return true;
}

ViewArg ReadView(u64 view) {
    ViewArg a{};
    if (view) a.view = Load<TextureView>(view), a.present = 1;
    return a;
}

// --- Texture -----------------------------------------------------------------
// The host keeps the builder's settings, the storage and the GPU address (the
// pool's storage address + offset). NVN's Finalize does nothing, so the game
// may initialise a texture's storage again without finalising it: the host
// object found there is then freed.

namespace {

struct Texture : Object {
    static constexpr Type kType = Type::kTexture;
    TextureState state{};
    u64 debug_id = 0;
};

std::mutex g_textures_mu;
std::unordered_map<u64, Texture*> g_textures; // the game's NVNtexture -> host object
std::atomic<u64> g_texture_ids{0};

} // namespace

TextureState TextureStateOf(u64 texture) { return Get<Texture>(texture)->state; }

NVN_PROC(nvnTextureInitialize, bool, u64 texture, u64 builder) {
    const auto b = Load<TextureBuilder>(builder);
    if (b.gl_name) Fatal("nvn: textures from GL texture names are not supported");
    TextureState st{};
    st.settings = b;
    st.storage = StorageOrStop(builder);
    const PoolRange p = PoolOf(b.pool);
    if (b.offset < 0 || static_cast<u64>(b.offset) + st.storage.size > p.size) {
        Log("nvn: nvnTextureInitialize: storage 0x%llx+0x%llx outside its pool (0x%llx bytes)",
            (unsigned long long)b.offset, (unsigned long long)st.storage.size, (unsigned long long)p.size);
    }
    st.address = p.storage + static_cast<u64>(b.offset);
    std::lock_guard lock(g_textures_mu);
    if (auto it = g_textures.find(texture); it != g_textures.end()) delete it->second;
    Texture* t = Create<Texture>(texture);
    t->state = st;
    t->debug_id = ++g_texture_ids;
    g_textures[texture] = t;
    return true;
}
NVN_PROC(nvnTextureFinalize, void, u64 texture) {
    std::lock_guard lock(g_textures_mu);
    g_textures.erase(texture);
    Destroy<Texture>(texture);
}
NVN_PROC(nvnTextureSetDebugLabel, void, u64 texture, u64 label) { Get<Texture>(texture); }
#define TX_GET(fn, type, expr)                                                                                      \
    NVN_PROC(fn, type, u64 texture) {                                                                               \
        const TextureState& s = Get<Texture>(texture)->state;                                                       \
        return static_cast<type>(expr);                                                                             \
    }
TX_GET(nvnTextureGetFlags, u32, s.settings.flags)
TX_GET(nvnTextureGetTarget, u32, s.settings.target)
TX_GET(nvnTextureGetWidth, u32, s.settings.width)
TX_GET(nvnTextureGetHeight, u32, s.settings.height)
TX_GET(nvnTextureGetDepth, u32, s.settings.depth)
TX_GET(nvnTextureGetLevels, u32, s.settings.levels)
TX_GET(nvnTextureGetFormat, u32, s.settings.format)
TX_GET(nvnTextureGetSamples, u32, s.settings.samples)
TX_GET(nvnTextureGetDepthStencilMode, u32, s.settings.ds_mode)
TX_GET(nvnTextureGetStride, s64, s.settings.stride)
TX_GET(nvnTextureGetStorageSize, u64, s.storage.size)

namespace {
// nvnTextureGetZCullStorageSize (BRO 0x577740 -> 0x576ba0 -> 0x5a5bb0): with the device's ZCull values
// (measured on the Switch in tap mode, 2026-10-16: alignment 32 x 32, 0x400 per unit, 32 bytes per unit
// plus 0x20 + 0xc0) the size of a single-sample depth texture is 32 x ceil(rup(w, 32) x rup(h, 32) / 1024)
// x depth + 224: 5 of 5 driver answers of the title route. Other formats and multisample take paths of
// the SDK not measured yet.
std::optional<u64> ZCullStorageSize(const TextureBuilder& b, std::string& why) {
    const FormatInfo* f = FindFormat(b.format);
    if (!f || !f->depth_stencil || b.samples > 1) {
        why = "format 0x" + std::to_string(b.format) + ", " + std::to_string(b.samples) + " samples";
        return std::nullopt;
    }
    auto rup = [](u64 v, u64 a) { return (v + a - 1) / a * a; };
    const u64 units = (rup(b.width, 32) * rup(b.height, 32) + 1023) / 1024;
    return 32 * units * std::max<u64>(b.depth, 1) + 0x20 + 0xc0;
}
} // namespace

namespace {
/// The byte offset of the view's first level and layer in the texture's storage (nvnTextureGetViewOffset).
std::optional<u64> ViewOffset(const TextureBuilder& b, u32 level, u32 layer, std::string& why) {
    const auto l = ComputeLayout(b, why);
    if (!l) return std::nullopt;
    if (b.target == kTarget3D && layer > 0) {
        // a 3D slice of level 0: slices are grouped by the block depth; inside a block, slice z starts after z
        // stacks of 2^bh GOBs (the sparse 3D block-height rule above; 4 of 4 driver answers)
        const FormatInfo* f = FindFormat(b.format);
        if (level != 0 || !f) return why = "a 3D slice offset of a level > 0", std::nullopt;
        const u64 wb = CeilDiv(b.width, f->block_w), hb = CeilDiv(b.height, f->block_h);
        const u32 bytes = f->bytes;
        const u64 tile_w = bytes == 1 || bytes == 2 ? 256 : bytes == 4 || bytes == 8 ? 128 : 64;
        const u64 tile_h = bytes == 1 ? 256 : bytes == 2 || bytes == 4 ? 128 : 64;
        if ((b.flags & kFlagSparse) && wb >= tile_w && hb >= tile_h) return why = "a 3D slice in sparse tiles", std::nullopt;
        const u64 gx = CeilDiv(wb * bytes, 64), gy = CeilDiv(hb, 8);
        u32 bh = (b.flags & kFlagSparse) ? std::min(BlockHeightLog2(hb), 2u) : 0, bd = BlockDepthLog2(b.depth);
        while (bh > 0 && gy <= (1ULL << (bh - 1))) --bh;
        while (bd > 0 && b.depth <= (1u << (bd - 1))) --bd;
        const u64 plane = gx * CeilDiv(gy, 1ULL << bh) * (512ULL << (bh + bd));
        return u64{layer >> bd} * plane + u64{layer & ((1u << bd) - 1)} * (512ULL << bh);
    }
    if (level >= l->level_offsets.size()) {
        why = "view level " + std::to_string(level) + " of " + std::to_string(l->level_offsets.size());
        return std::nullopt;
    }
    return u64{layer} * l->layer_size + l->level_offsets[level];
}
} // namespace

NVN_PROC(nvnTextureGetViewOffset, s64, u64 texture, u64 view) {
    const TextureState t = TextureStateOf(texture);
    const ViewArg v = ReadView(view);
    const u32 level = v.present && (v.view.set & kViewLevels) ? v.view.base_level : 0;
    const u32 layer = v.present && (v.view.set & kViewLayers) ? static_cast<u32>(v.view.base_layer) : 0;
    std::string why;
    const auto off = ViewOffset(t.settings, level, layer, why);
    if (!off) Fatal("nvn: nvnTextureGetViewOffset: %s", why.c_str());
    return static_cast<s64>(*off);
}

NVN_PROC(nvnTextureGetZCullStorageSize, u64, u64 texture) {
    std::string why;
    const auto size = ZCullStorageSize(TextureStateOf(texture).settings, why);
    if (!size) Fatal("nvn: nvnTextureGetZCullStorageSize for %s is not implemented yet", why.c_str());
    return *size;
}
TX_GET(nvnTextureGetMemoryPool, u64, s.settings.pool)
TX_GET(nvnTextureGetMemoryOffset, s64, s.settings.offset)
TX_GET(nvnTextureGetTextureAddress, u64, s.address)
#undef TX_GET
NVN_PROC(nvnTextureGetDebugID, u64, u64 texture) { return Get<Texture>(texture)->debug_id; }
NVN_PROC(nvnTextureGetSwizzle, void, u64 texture, u64 r, u64 g, u64 b, u64 a) {
    const TextureState& s = Get<Texture>(texture)->state;
    hle::WriteGuest<u32>(r, s.settings.swizzle[0]);
    hle::WriteGuest<u32>(g, s.settings.swizzle[1]);
    hle::WriteGuest<u32>(b, s.settings.swizzle[2]);
    hle::WriteGuest<u32>(a, s.settings.swizzle[3]);
}

// --- Pool registration ---------------------------------------------------------------

namespace {
void RegisterInPool(u64 pool, s32 id, u64 texture, u64 view, bool image) {
    TexturePool* o = Get<TexturePool>(pool);
    if (id < 0 || id >= o->size) Fatal("nvn: texture id %d outside its pool (%d entries)", id, o->size);
    o->entries[static_cast<size_t>(id)] = PoolEntry{TextureStateOf(texture), ReadView(view), image, true};
}
} // namespace

std::optional<PoolTexture> TexturePoolEntry(u64 pool, u32 id) {
    const auto* o = static_cast<const TexturePool*>(FindObject(pool, Type::kTexturePool));
    if (!o || id >= static_cast<u32>(o->size) || !o->entries[id].registered) return std::nullopt;
    const PoolEntry& e = o->entries[id];
    return PoolTexture{e.texture, e.view, e.image};
}

NVN_PROC(nvnTexturePoolRegisterTexture, void, u64 pool, s32 id, u64 texture, u64 view) {
    RegisterInPool(pool, id, texture, view, false);
}
NVN_PROC(nvnTexturePoolRegisterImage, void, u64 pool, s32 id, u64 texture, u64 view) {
    RegisterInPool(pool, id, texture, view, true);
}

// --- Copies into textures (records; translated with Phase 6.3) ---------------------------------------

NVN_PROC(nvnCommandBufferCopyBufferToTexture, void, u64 cmdbuf, u64 buffer, u64 texture, u64 view, u64 region,
         u32 flags) {
    CopyBufferToTextureRecord r{};
    r.buffer = buffer;
    r.texture = TextureStateOf(texture);
    r.view = ReadView(view);
    r.region = hle::ReadGuest<CopyRegion>(region);
    r.flags = flags;
    Record(c, cmdbuf, Op::kCopyBufferToTexture, r);
}

// ClearTexture, ClearTexturei, ClearTextureui (cmdbuf, texture, view, region, colour[4], mask): the same
// record; the region as for copies (BRO 0x592a30: an empty region records nothing).
#define NVN_CLEAR_TEXTURE(fn)                                                                                       \
    NVN_PROC(fn, void, u64 cmdbuf, u64 texture, u64 view, u64 region, u64 color, u32 mask) {                       \
        ClearTextureRecord r{};                                                                                     \
        r.region = hle::ReadGuest<CopyRegion>(region);                                                              \
        if (r.region.width == 0 || r.region.height == 0 || r.region.depth == 0) return;                             \
        r.texture = TextureStateOf(texture);                                                                        \
        r.view = ReadView(view);                                                                                    \
        const auto words = hle::ReadGuest<std::array<u32, 4>>(color);                                               \
        std::memcpy(r.color, words.data(), sizeof r.color);                                                         \
        r.mask = mask;                                                                                              \
        Record(c, cmdbuf, Op::kClearTexture, r);                                                                    \
    }
NVN_CLEAR_TEXTURE(nvnCommandBufferClearTexture)
NVN_CLEAR_TEXTURE(nvnCommandBufferClearTexturei)
NVN_CLEAR_TEXTURE(nvnCommandBufferClearTextureui)
#undef NVN_CLEAR_TEXTURE

// CopyTextureToBuffer(texture, view, region, buffer, flags) (BRO 0x565c60): the same record as the copy in.
NVN_PROC(nvnCommandBufferCopyTextureToBuffer, void, u64 cmdbuf, u64 texture, u64 view, u64 region, u64 buffer,
         u32 flags) {
    CopyBufferToTextureRecord r{};
    r.buffer = buffer;
    r.texture = TextureStateOf(texture);
    r.view = ReadView(view);
    r.region = hle::ReadGuest<CopyRegion>(region);
    r.flags = flags;
    Record(c, cmdbuf, Op::kCopyTextureToBuffer, r);
}

// CopyTextureToTexture(src, src view, src region, dst, dst view, dst region, flags) (BRO 0x566110).
NVN_PROC(nvnCommandBufferCopyTextureToTexture, void, u64 cmdbuf, u64 src, u64 src_view, u64 src_region, u64 dst,
         u64 dst_view, u64 dst_region, u32 flags) {
    CopyTextureToTextureRecord r{};
    r.src = TextureStateOf(src);
    r.src_view = ReadView(src_view);
    r.src_region = hle::ReadGuest<CopyRegion>(src_region);
    r.dst = TextureStateOf(dst);
    r.dst_view = ReadView(dst_view);
    r.dst_region = hle::ReadGuest<CopyRegion>(dst_region);
    r.flags = flags;
    Record(c, cmdbuf, Op::kCopyTextureToTexture, r);
}

// --- Tap mode: the real driver's storage sizes ---------------------------
// After each nvnTextureBuilderGetStorageSize / GetStorageAlignment of the real
// driver, the builder's settings are read back through the SDK's own getters
// and each new combination is appended, with the driver's answer, to
// nvn_texture_sizes.tsv in the working directory: the ground truth the
// layer's layout code is checked against. A combination that gets two
// different answers is logged (the getters would then miss an input).

namespace {

struct BuilderParams {
    u32 target, width, height, depth, levels, format, flags, samples, stride, ds_mode;
    u64 layout; // NVNpackagedTextureLayout (8 bytes)
    auto Key() const {
        return std::tie(target, width, height, depth, levels, format, flags, samples, stride, ds_mode, layout);
    }
    bool operator<(const BuilderParams& o) const { return Key() < o.Key(); }
};

BuilderParams ReadRealBuilder(GuestContext* c, u64 builder) {
    auto get = [&](const char* fn) {
        return static_cast<u32>(hle::CallGuest(c, RealProc(c, fn), {builder}, true));
    };
    BuilderParams p{};
    p.target = get("nvnTextureBuilderGetTarget");
    p.width = get("nvnTextureBuilderGetWidth");
    p.height = get("nvnTextureBuilderGetHeight");
    p.depth = get("nvnTextureBuilderGetDepth");
    p.levels = get("nvnTextureBuilderGetLevels");
    p.format = get("nvnTextureBuilderGetFormat");
    p.flags = get("nvnTextureBuilderGetFlags");
    p.samples = get("nvnTextureBuilderGetSamples");
    p.stride = get("nvnTextureBuilderGetStride");
    p.ds_mode = get("nvnTextureBuilderGetDepthStencilMode");
    auto lock = LockScratch();
    hle::WriteGuest<u64>(Scratch::kOut, 0);
    hle::CallGuest(c, RealProc(c, "nvnTextureBuilderGetPackagedTextureLayout"), {builder, Scratch::kOut}, true);
    p.layout = hle::ReadGuest<u64>(Scratch::kOut);
    return p;
}

/// The layer's answer for the same settings against the driver's; logs each
/// difference (up to 20) and the counts at 1, 10, 100 ... calls.
void CompareWithModel(const std::string& fn, bool alignment, const BuilderParams& p, u64 driver,
                      bool low32 = false) {
    static std::mutex mu;
    static u64 checked = 0, different = 0, unhandled = 0;
    TextureBuilder b{};
    b.target = p.target, b.width = p.width, b.height = p.height, b.depth = p.depth, b.levels = p.levels;
    b.format = p.format, b.flags = p.flags, b.samples = p.samples, b.stride = static_cast<s32>(p.stride);
    b.ds_mode = p.ds_mode, b.layout = p.layout, b.layout_set = p.layout != 0;
    std::string why;
    const auto s = ComputeStorage(b, why);
    std::lock_guard lock(mu);
    ++checked;
    if (!s) {
        if (++unhandled <= 20) Log("nvn texture oracle: not handled by the layer: %s", why.c_str());
    } else if ((alignment ? s->alignment : low32 ? static_cast<u32>(s->size) : s->size) != driver &&
               ++different <= 20) {
        Log("nvn texture oracle: %s DIFFERENT: target %u %ux%ux%u levels %u format 0x%x flags 0x%x: driver 0x%llx, "
            "layer 0x%llx",
            fn.c_str(), p.target, p.width, p.height, p.depth, p.levels, p.format, p.flags, (unsigned long long)driver,
            (unsigned long long)(alignment ? s->alignment : s->size));
    }
    u64 n = checked;
    while (n % 10 == 0) n /= 10;
    if (n == 1) {
        Log("nvn texture oracle: %llu answers checked, %llu different, %llu not handled", (unsigned long long)checked,
            (unsigned long long)different, (unsigned long long)unhandled);
    }
}

std::mutex g_sizes_mu;
std::map<std::pair<bool, BuilderParams>, u64> g_sizes; // (alignment?, settings) -> driver's answer
FILE* g_sizes_file = nullptr;

void ShadowStorage(const std::string& fn, const GuestContext& before, const GuestContext& after) {
    const bool alignment = fn == "nvnTextureBuilderGetStorageAlignment";
    const BuilderParams p = ReadRealBuilder(CurrentContext(), before.x[0]);
    const u64 driver = alignment ? static_cast<u32>(after.x[0]) : after.x[0];
    CompareWithModel(fn, alignment, p, driver);
    std::lock_guard lock(g_sizes_mu);
    auto [it, fresh] = g_sizes.emplace(std::make_pair(alignment, p), driver);
    if (!fresh) {
        if (it->second != driver) {
            Log("nvn texture oracle: %s gives 0x%llx and 0x%llx for the same settings (target %u, %ux%ux%u, format 0x%x)",
                fn.c_str(), (unsigned long long)it->second, (unsigned long long)driver, p.target, p.width, p.height,
                p.depth, p.format);
        }
        return;
    }
    if (!g_sizes_file) {
        g_sizes_file = std::fopen("nvn_texture_sizes.tsv", "w");
        if (!g_sizes_file) return;
        std::fprintf(g_sizes_file, "query\ttarget\twidth\theight\tdepth\tlevels\tformat\tflags\tsamples\tstride\t"
                                   "ds_mode\tlayout\tdriver\n");
    }
    std::fprintf(g_sizes_file, "%s\t%u\t%u\t%u\t%u\t%u\t0x%x\t0x%x\t%u\t%u\t%u\t0x%llx\t0x%llx\n",
                 alignment ? "alignment" : "size", p.target, p.width, p.height, p.depth, p.levels, p.format, p.flags,
                 p.samples, p.stride, p.ds_mode, (unsigned long long)p.layout, (unsigned long long)driver);
    std::fflush(g_sizes_file);
}

/// After each real nvnTextureInitialize: the texture's storage size as the
/// driver answers it (nvnTextureGetStorageSize, 32 bits in SDK 12.3.2) against
/// the layer's for the same builder. Covers every texture the game creates,
/// buffer textures included (the game never asks their builders' size).
void ShadowTextureInit(const std::string&, const GuestContext& before, const GuestContext& after) {
    if (!(after.x[0] & 0xFF)) return;
    GuestContext* c = CurrentContext();
    const BuilderParams p = ReadRealBuilder(c, before.x[1]);
    const u64 driver =
        static_cast<u32>(hle::CallGuest(c, RealProc(c, "nvnTextureGetStorageSize"), {before.x[0]}, true));
    CompareWithModel("nvnTextureGetStorageSize", false, p, driver, true);
}

/// After each real nvnTextureGetZCullStorageSize: the texture's settings (through the SDK's getters), the
/// driver's answer and, once, the device words its formula reads (+0xd8..+0xfc; BRO 0x577740 / 0x576ba0),
/// as "nvn zcull:" lines — the data for the layer's own answer (measure first).
void ShadowZCullSize(const std::string&, const GuestContext& before, const GuestContext& after) {
    static std::once_flag device_once;
    GuestContext* c = CurrentContext();
    const u64 texture = before.x[0];
    auto get = [&](const char* fn) { return static_cast<u32>(hle::CallGuest(c, RealProc(c, fn), {texture}, true)); };
    Log("nvn zcull: target %u %ux%ux%u format 0x%x samples %u flags 0x%x levels %u: driver 0x%llx", get("nvnTextureGetTarget"),
        get("nvnTextureGetWidth"), get("nvnTextureGetHeight"), get("nvnTextureGetDepth"), get("nvnTextureGetFormat"),
        get("nvnTextureGetSamples"), get("nvnTextureGetFlags"), get("nvnTextureGetLevels"), (unsigned long long)after.x[0]);
    TextureBuilder tb{};
    tb.target = get("nvnTextureGetTarget"), tb.width = get("nvnTextureGetWidth"), tb.height = get("nvnTextureGetHeight");
    tb.depth = get("nvnTextureGetDepth"), tb.format = get("nvnTextureGetFormat"), tb.samples = get("nvnTextureGetSamples");
    std::string why;
    const auto ours = ZCullStorageSize(tb, why);
    if (!ours) Log("nvn zcull: not handled by the layer: %s", why.c_str());
    else if (*ours != after.x[0]) Log("nvn zcull: DIFFERENT: layer 0x%llx", (unsigned long long)*ours);
    std::call_once(device_once, [&] {
        const u64 device = hle::ReadGuest<u64>(texture + 8);
        std::string words;
        for (u32 off = 0xd8; off < 0x100; off += 4) {
            char b[32];
            std::snprintf(b, sizeof b, " +0x%x=0x%x", off, hle::ReadGuest<u32>(device + off));
            words += b;
        }
        Log("nvn zcull: device 0x%llx words:%s", (unsigned long long)device, words.c_str());
    });
}

/// After each real nvnTextureGetViewOffset: the layer's answer for the same texture and view.
void ShadowViewOffset(const std::string&, const GuestContext& before, const GuestContext& after) {
    static std::mutex mu;
    static u64 checked = 0, different = 0, unhandled = 0;
    GuestContext* c = CurrentContext();
    const u64 texture = before.x[0], view = before.x[1];
    auto get = [&](const char* fn) { return static_cast<u32>(hle::CallGuest(c, RealProc(c, fn), {texture}, true)); };
    TextureBuilder tb{};
    tb.target = get("nvnTextureGetTarget"), tb.width = get("nvnTextureGetWidth"), tb.height = get("nvnTextureGetHeight");
    tb.depth = get("nvnTextureGetDepth"), tb.levels = get("nvnTextureGetLevels"), tb.format = get("nvnTextureGetFormat");
    tb.flags = get("nvnTextureGetFlags"), tb.samples = get("nvnTextureGetSamples");
    tb.stride = static_cast<s32>(get("nvnTextureGetStride"));
    u32 level = 0, layer = 0;
    if (view) {
        auto lock = LockScratch();
        auto pair = [&](const char* fn, u32& first) {
            hle::WriteGuest<std::array<s32, 2>>(Scratch::kOut, {0, 0});
            if (hle::CallGuest(c, RealProc(c, fn), {view, Scratch::kOut, Scratch::kOut + 4}, true) & 0xFF) {
                first = static_cast<u32>(hle::ReadGuest<s32>(Scratch::kOut));
            }
        };
        pair("nvnTextureViewGetLevels", level);
        pair("nvnTextureViewGetLayers", layer);
    }
    std::string why;
    const auto ours = ViewOffset(tb, level, layer, why);
    std::lock_guard lock(mu);
    ++checked;
    if (!ours) {
        if (++unhandled <= 20) Log("nvn view offset: not handled by the layer: %s", why.c_str());
    } else if (*ours != after.x[0] && ++different <= 20) {
        Log("nvn view offset: DIFFERENT: target %u %ux%ux%u levels %u format 0x%x flags 0x%x, level %u layer %u: "
            "driver 0x%llx, layer 0x%llx", tb.target, tb.width, tb.height, tb.depth, tb.levels, tb.format, tb.flags,
            level, layer, (unsigned long long)after.x[0], (unsigned long long)*ours);
    }
    u64 n = checked;
    while (n % 10 == 0) n /= 10;
    if (n == 1) {
        Log("nvn view offset: %llu answers checked, %llu different, %llu not handled", (unsigned long long)checked,
            (unsigned long long)different, (unsigned long long)unhandled);
    }
}

} // namespace

void InstallTextureShadows() {
    tap::AddShadow("nvnTextureBuilderGetStorageSize", ShadowStorage);
    tap::AddShadow("nvnTextureBuilderGetStorageAlignment", ShadowStorage);
    tap::AddShadow("nvnTextureInitialize", ShadowTextureInit);
    tap::AddShadow("nvnTextureGetZCullStorageSize", ShadowZCullSize);
    tap::AddShadow("nvnTextureGetViewOffset", ShadowViewOffset);
}

} // namespace rt::nvn
