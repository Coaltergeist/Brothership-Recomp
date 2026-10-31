// NVNsamplerBuilder, NVNsampler and NVNsamplerPool (NVN_PLAN 6.5, 6.6: the
// state only; Vulkan samplers come with submission).
//
// A sampler builder holds its fields inline in the game's storage, in 48
// bytes as NVN's own (SDK 12.3.2: device, LOD clamp, bias, anisotropy, LOD
// snap, border colour, one word of enums), in the layer's layout. A sampler
// copies the builder's state to the host. A sampler pool is a host table of
// sampler states by id; NVN also writes hardware descriptors into the pool's
// memory, which only the GPU reads, so the layer leaves that memory alone.
#include "vk.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace rt::nvn {

namespace {

struct SamplerState { // inline in NVNsamplerBuilder; copied by NVNsampler
    u64 device;
    float lod_min, lod_max, lod_bias, max_anisotropy, lod_snap;
    u32 border[4]; // float, int or uint bits, as set
    // NVN's enums, 4 bits each (every value NVN defines fits)
    u32 wrap_s : 4, wrap_t : 4, wrap_r : 4, reduction : 4, min_filter : 4, mag_filter : 4, compare_mode : 4,
        compare_func : 4;
};
static_assert(sizeof(SamplerState) == 48);

/// Stops the run for an enum value outside 0..15 (no NVN value is).
u32 Field(const char* fn, u32 v) {
    if (v > 15) Fatal("nvn: %s: value %u is no NVN enum value the layer knows", fn, v);
    return v;
}

struct Sampler : Object {
    static constexpr Type kType = Type::kSampler;
    SamplerState s{};
    u64 debug_id = 0;
};

struct SamplerPool : Object {
    static constexpr Type kType = Type::kSamplerPool;
    u64 memory_pool = 0;
    s64 offset = 0;
    s32 size = 0;
    std::vector<SamplerState> entries;
    std::vector<bool> registered;
};

std::atomic<u64> g_debug_ids{0};

/// The builder's state, changed by `f`.
template <typename F>
void Change(u64 builder, F&& f) {
    auto s = Load<SamplerState>(builder);
    f(s);
    Store(builder, s);
}

} // namespace

// --- SamplerBuilder -----------------------------------------------------------------------------

NVN_PROC(nvnSamplerBuilderSetDevice, void, u64 builder, u64 device) {
    Change(builder, [&](SamplerState& s) { s.device = device; });
}
NVN_PROC(nvnSamplerBuilderSetDefaults, void, u64 builder) {
    Change(builder, [](SamplerState& s) {
        const u64 device = s.device;
        s = SamplerState{};
        s.device = device;
        s.lod_min = 0.0f;
        s.lod_max = 1000.0f;
        s.max_anisotropy = 1.0f;
        s.wrap_s = s.wrap_t = s.wrap_r = 1;
        s.min_filter = s.mag_filter = 1;
        s.compare_func = 2;
    });
}
NVN_PROC(nvnSamplerBuilderSetMinMagFilter, void, u64 builder, u32 min, u32 mag) {
    Change(builder, [&](SamplerState& s) {
        s.min_filter = Field("nvnSamplerBuilderSetMinMagFilter", min);
        s.mag_filter = Field("nvnSamplerBuilderSetMinMagFilter", mag);
    });
}
NVN_PROC(nvnSamplerBuilderSetWrapMode, void, u64 builder, u32 s_, u32 t, u32 r) {
    Change(builder, [&](SamplerState& s) {
        s.wrap_s = Field("nvnSamplerBuilderSetWrapMode", s_);
        s.wrap_t = Field("nvnSamplerBuilderSetWrapMode", t);
        s.wrap_r = Field("nvnSamplerBuilderSetWrapMode", r);
    });
}
NVN_PROC(nvnSamplerBuilderSetLodClamp, void, u64 builder, float min, float max) {
    Change(builder, [&](SamplerState& s) {
        s.lod_min = min;
        s.lod_max = max;
    });
}
NVN_PROC(nvnSamplerBuilderSetLodBias, void, u64 builder, float bias) {
    Change(builder, [&](SamplerState& s) { s.lod_bias = bias; });
}
NVN_PROC(nvnSamplerBuilderSetCompare, void, u64 builder, u32 mode, u32 func) {
    Change(builder, [&](SamplerState& s) {
        s.compare_mode = Field("nvnSamplerBuilderSetCompare", mode);
        s.compare_func = Field("nvnSamplerBuilderSetCompare", func);
    });
}
NVN_PROC(nvnSamplerBuilderSetMaxAnisotropy, void, u64 builder, float max) {
    Change(builder, [&](SamplerState& s) { s.max_anisotropy = max; });
}
NVN_PROC(nvnSamplerBuilderSetReductionFilter, void, u64 builder, u32 filter) {
    Change(builder, [&](SamplerState& s) { s.reduction = Field("nvnSamplerBuilderSetReductionFilter", filter); });
}
NVN_PROC(nvnSamplerBuilderSetLodSnap, void, u64 builder, float snap) {
    Change(builder, [&](SamplerState& s) { s.lod_snap = snap; });
}
// The three border colour setters store the same four words, as NVN.
#define NVN_SAMPLER_BORDER_SET(fn)                                                                                  \
    NVN_PROC(fn, void, u64 builder, u64 colour) {                                                                   \
        const auto v = hle::ReadGuest<std::array<u32, 4>>(colour);                                                  \
        Change(builder, [&](SamplerState& s) { std::memcpy(s.border, v.data(), sizeof s.border); });               \
    }
NVN_SAMPLER_BORDER_SET(nvnSamplerBuilderSetBorderColor)
NVN_SAMPLER_BORDER_SET(nvnSamplerBuilderSetBorderColori)
NVN_SAMPLER_BORDER_SET(nvnSamplerBuilderSetBorderColorui)
#undef NVN_SAMPLER_BORDER_SET

NVN_PROC(nvnSamplerBuilderGetDevice, u64, u64 builder) { return Load<SamplerState>(builder).device; }

// --- Getters of builders and samplers ------------------------------------------------------------

namespace {
SamplerState StateOf(u64 obj, bool sampler) { return sampler ? Get<Sampler>(obj)->s : Load<SamplerState>(obj); }
} // namespace

// Each getter twice: nvnSamplerBuilderGetX (inline state) and nvnSamplerGetX (host state).
#define NVN_SAMPLER_GETTERS(prefix, is_sampler)                                                                     \
    NVN_PROC(prefix##GetMinMagFilter, void, u64 obj, u64 min, u64 mag) {                                            \
        const auto s = StateOf(obj, is_sampler);                                                                    \
        hle::WriteGuest<u32>(min, s.min_filter);                                                                    \
        hle::WriteGuest<u32>(mag, s.mag_filter);                                                                    \
    }                                                                                                               \
    NVN_PROC(prefix##GetWrapMode, void, u64 obj, u64 s_, u64 t, u64 r) {                                            \
        const auto s = StateOf(obj, is_sampler);                                                                    \
        hle::WriteGuest<u32>(s_, s.wrap_s);                                                                         \
        hle::WriteGuest<u32>(t, s.wrap_t);                                                                          \
        hle::WriteGuest<u32>(r, s.wrap_r);                                                                          \
    }                                                                                                               \
    NVN_PROC(prefix##GetLodClamp, void, u64 obj, u64 min, u64 max) {                                                \
        const auto s = StateOf(obj, is_sampler);                                                                    \
        hle::WriteGuest<float>(min, s.lod_min);                                                                     \
        hle::WriteGuest<float>(max, s.lod_max);                                                                     \
    }                                                                                                               \
    NVN_PROC(prefix##GetLodBias, float, u64 obj) { return StateOf(obj, is_sampler).lod_bias; }                      \
    NVN_PROC(prefix##GetCompare, void, u64 obj, u64 mode, u64 func) {                                               \
        const auto s = StateOf(obj, is_sampler);                                                                    \
        hle::WriteGuest<u32>(mode, s.compare_mode);                                                                 \
        hle::WriteGuest<u32>(func, s.compare_func);                                                                 \
    }                                                                                                               \
    NVN_PROC(prefix##GetMaxAnisotropy, float, u64 obj) { return StateOf(obj, is_sampler).max_anisotropy; }          \
    NVN_PROC(prefix##GetReductionFilter, u32, u64 obj) { return StateOf(obj, is_sampler).reduction; }               \
    NVN_PROC(prefix##GetBorderColor, void, u64 obj, u64 out) {                                                      \
        hle::WriteGuest(out, StateOf(obj, is_sampler).border);                                                      \
    }                                                                                                               \
    NVN_PROC(prefix##GetBorderColori, void, u64 obj, u64 out) {                                                     \
        hle::WriteGuest(out, StateOf(obj, is_sampler).border);                                                      \
    }                                                                                                               \
    NVN_PROC(prefix##GetBorderColorui, void, u64 obj, u64 out) {                                                    \
        hle::WriteGuest(out, StateOf(obj, is_sampler).border);                                                      \
    }
NVN_SAMPLER_GETTERS(nvnSamplerBuilder, false)
NVN_SAMPLER_GETTERS(nvnSampler, true)
#undef NVN_SAMPLER_GETTERS

NVN_PROC(nvnSamplerBuilderGetLodSnap, float, u64 builder) { return Load<SamplerState>(builder).lod_snap; }

// --- Sampler -------------------------------------------------------------------------------------

NVN_PROC(nvnSamplerInitialize, bool, u64 sampler, u64 builder) {
    Sampler* o = Create<Sampler>(sampler);
    o->s = Load<SamplerState>(builder);
    o->debug_id = ++g_debug_ids;
    return true;
}
NVN_PROC(nvnSamplerFinalize, void, u64 sampler) { Destroy<Sampler>(sampler); }
NVN_PROC(nvnSamplerSetDebugLabel, void, u64 sampler, u64 label) { Get<Sampler>(sampler); }
NVN_PROC(nvnSamplerGetDebugID, u64, u64 sampler) { return Get<Sampler>(sampler)->debug_id; }

// --- SamplerPool ---------------------------------------------------------------------------------

constexpr u64 kDescriptorSize = 32; // NVN's, per entry of the pool's memory

NVN_PROC(nvnSamplerPoolInitialize, bool, u64 pool, u64 memory_pool, s64 offset, s32 count) {
    const PoolRange p = PoolOf(memory_pool);
    if (offset < 0 || count < 0 || static_cast<u64>(offset) + kDescriptorSize * static_cast<u64>(count) > p.size) {
        Fatal("nvn: nvnSamplerPoolInitialize: %d descriptors at 0x%llx do not fit the memory pool (0x%llx bytes)",
              count, (unsigned long long)offset, (unsigned long long)p.size);
    }
    SamplerPool* o = Create<SamplerPool>(pool);
    o->memory_pool = memory_pool;
    o->offset = offset;
    o->size = count;
    o->entries.resize(static_cast<size_t>(count));
    o->registered.resize(static_cast<size_t>(count));
    return true;
}
NVN_PROC(nvnSamplerPoolFinalize, void, u64 pool) { Destroy<SamplerPool>(pool); }
NVN_PROC(nvnSamplerPoolSetDebugLabel, void, u64 pool, u64 label) { Get<SamplerPool>(pool); }
NVN_PROC(nvnSamplerPoolGetMemoryPool, u64, u64 pool) { return Get<SamplerPool>(pool)->memory_pool; }
NVN_PROC(nvnSamplerPoolGetMemoryOffset, s64, u64 pool) { return Get<SamplerPool>(pool)->offset; }
NVN_PROC(nvnSamplerPoolGetSize, s32, u64 pool) { return Get<SamplerPool>(pool)->size; }

namespace {
void Register(u64 pool, s32 id, const SamplerState& s) {
    SamplerPool* o = Get<SamplerPool>(pool);
    if (id < 0 || id >= o->size) Fatal("nvn: sampler id %d outside its pool (%d entries)", id, o->size);
    o->entries[static_cast<size_t>(id)] = s;
    o->registered[static_cast<size_t>(id)] = true;
}
} // namespace

NVN_PROC(nvnSamplerPoolRegisterSampler, void, u64 pool, s32 id, u64 sampler) {
    Register(pool, id, Get<Sampler>(sampler)->s);
}
NVN_PROC(nvnSamplerPoolRegisterSamplerBuilder, void, u64 pool, s32 id, u64 builder) {
    Register(pool, id, Load<SamplerState>(builder));
}

// --- Vulkan samplers (NVN_PLAN 7.5) ---------------------------------------------------------------------

namespace {

// NVN's enums (as LASRecomp's runtime/nvn/vulkan/textures.c, MIT; the defaults above agree): min filter 0
// nearest, 1 linear, 2-5 nearest/linear x mipmap nearest/linear; wrap 0 clamp, 1 repeat, 2/3 mirror clamp
// (to edge), 4/5 (mirror) clamp to border, 6 mirrored repeat, 7 clamp to edge; compare functions as depth
// tests (1 never ... 8 always).
VkSamplerAddressMode Wrap(u32 w) {
    switch (w) {
    case 0:
    case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 2:
    case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    case 4:
    case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    }
    Fatal("nvn: sampler wrap mode %u", w);
}

bool Anisotropy() {
    static const bool on = Vk().device->GetPhysical().GetFeatures().samplerAnisotropy == VK_TRUE;
    return on;
}

std::mutex g_vk_samplers_mu;
std::map<std::array<u8, sizeof(SamplerState)>, Vulkan::vk::Sampler> g_vk_samplers;

} // namespace

VkSampler SamplerFor(u64 pool, u32 id) {
    const auto* o = static_cast<const SamplerPool*>(FindObject(pool, Type::kSamplerPool));
    if (!o) Fatal("nvn: a draw samples with sampler %u, but no sampler pool is set", id);
    SamplerState st{};
    if (id < static_cast<u32>(o->size) && o->registered[id]) {
        st = o->entries[id];
    } else { // a texture only fetched from (texel fetches ignore the sampler): Vulkan still wants one
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) Log("nvn: a draw names unregistered sampler %u: a default sampler stands in", id);
        st.min_filter = st.mag_filter = 0; // nearest, clamp to edge (wrap 0)
        st.lod_max = 1000.0f;
        st.max_anisotropy = 1.0f;
        st.compare_func = 2;
    }
    st.device = 0; // not part of the sampler
    std::array<u8, sizeof st> key;
    std::memcpy(key.data(), &st, sizeof st);
    std::lock_guard lock(g_vk_samplers_mu);
    auto& smp = g_vk_samplers[key];
    if (!smp) {
        if (st.min_filter > 5 || st.mag_filter > 1) Fatal("nvn: sampler filters %u/%u", st.min_filter, st.mag_filter);
        if (st.reduction != 0) Fatal("nvn: sampler reduction filter %u is not implemented yet", st.reduction);
        const bool border = st.wrap_s == 4 || st.wrap_s == 5 || st.wrap_t == 4 || st.wrap_t == 5 || st.wrap_r == 4 || st.wrap_r == 5;
        VkSamplerCustomBorderColorCreateInfoEXT custom{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
            .pNext = nullptr,
            .customBorderColor = {},
            .format = VK_FORMAT_UNDEFINED,
        };
        std::memcpy(custom.customBorderColor.uint32, st.border, sizeof st.border);
        const bool use_custom = border && Vk().device->IsCustomBorderColorsSupported();
        const float aniso = std::max(1.0f, st.max_anisotropy);
        const u32 mf = st.min_filter;
        smp = Vk().device->GetLogical().CreateSampler({
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = use_custom ? &custom : nullptr,
            .flags = 0,
            .magFilter = st.mag_filter == 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
            .minFilter = mf == 1 || mf == 3 || mf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
            .mipmapMode = mf == 4 || mf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = Wrap(st.wrap_s),
            .addressModeV = Wrap(st.wrap_t),
            .addressModeW = Wrap(st.wrap_r),
            .mipLodBias = st.lod_bias,
            .anisotropyEnable = aniso > 1.0f && Anisotropy() ? VK_TRUE : VK_FALSE,
            .maxAnisotropy = aniso,
            .compareEnable = st.compare_mode != 0 ? VK_TRUE : VK_FALSE,
            .compareOp = static_cast<VkCompareOp>(std::clamp<u32>(st.compare_func, 1, 8) - 1), // NVN 1 never .. 8 always
            .minLod = st.lod_min,
            // without mipmaps (min filter 0 / 1) Vulkan samples level 0 only when maxLod is 0.25 or less
            .maxLod = mf <= 1 ? 0.25f : st.lod_max,
            .borderColor = use_custom ? VK_BORDER_COLOR_FLOAT_CUSTOM_EXT : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        });
    }
    return *smp;
}

} // namespace rt::nvn
