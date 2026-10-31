// SHADER_PLAN Phase 4: the local store of translated stages (shader_store.h).
#include "shader_store.h"
#include "shaders.h"
#include "vk.h"

#include "../gpu.h"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>

#ifndef RT_DRIPPU_COMMIT
#define RT_DRIPPU_COMMIT "unknown"
#endif

namespace rt::nvn {

namespace {

// Bumped whenever our translation of a stage changes (TranslateStageInto, MakeCompute, their hooks, the SPIR-V
// passes after drippu): a stored entry must be what play would translate now.
constexpr u32 kStoreVersion = 3; // 2: RescalingPass; 3: its per-stage bit ranges (2026-10-31)
constexpr char kStoreMagic[8] = {'N', 'X', 'R', 'T', 'S', 'T', 'G', '1'};

// drippu's Shader::Info is written field by field (InfoFields); a field drippu adds or removes changes its size,
// which stops the build here (then: InfoFields and kStoreVersion).
#ifdef _WIN32
constexpr size_t kInfoSize = 2176; // clang-cl
#else
constexpr size_t kInfoSize = 2216; // GCC
#endif
static_assert(sizeof(Shader::Info) == kInfoSize, "drippu's Shader::Info changed: update InfoFields and kStoreVersion");

// drippu's descriptors have padding after their bool / enum fields, and a copy carries whatever bytes were there:
// written as they are, two equal translations could differ in the file (the store check found stages whose
// StorageBufferDescriptor padding differed, 2026-10-31). Out writes them rebuilt field by field in zeroed memory.
template <typename T>
T Zeroed() {
    T t;
    std::memset(&t, 0, sizeof t);
    return t;
}
const Shader::ConstantBufferDescriptor& Clean(const Shader::ConstantBufferDescriptor& d) { return d; } // no padding
Shader::StorageBufferDescriptor Clean(const Shader::StorageBufferDescriptor& d) {
    auto t = Zeroed<Shader::StorageBufferDescriptor>();
    t.cbuf_index = d.cbuf_index, t.cbuf_offset = d.cbuf_offset, t.count = d.count, t.is_written = d.is_written;
    return t;
}
Shader::TextureBufferDescriptor Clean(const Shader::TextureBufferDescriptor& d) {
    auto t = Zeroed<Shader::TextureBufferDescriptor>();
    t.has_secondary = d.has_secondary, t.cbuf_index = d.cbuf_index, t.cbuf_offset = d.cbuf_offset;
    t.shift_left = d.shift_left, t.secondary_cbuf_index = d.secondary_cbuf_index;
    t.secondary_cbuf_offset = d.secondary_cbuf_offset, t.secondary_shift_left = d.secondary_shift_left;
    t.count = d.count, t.size_shift = d.size_shift;
    return t;
}
Shader::ImageBufferDescriptor Clean(const Shader::ImageBufferDescriptor& d) {
    auto t = Zeroed<Shader::ImageBufferDescriptor>();
    t.format = d.format, t.is_written = d.is_written, t.is_read = d.is_read, t.is_integer = d.is_integer;
    t.cbuf_index = d.cbuf_index, t.cbuf_offset = d.cbuf_offset, t.count = d.count, t.size_shift = d.size_shift;
    return t;
}
Shader::TextureDescriptor Clean(const Shader::TextureDescriptor& d) {
    auto t = Zeroed<Shader::TextureDescriptor>();
    t.type = d.type, t.is_depth = d.is_depth, t.is_multisample = d.is_multisample, t.is_integer = d.is_integer;
    t.has_secondary = d.has_secondary, t.cbuf_index = d.cbuf_index, t.cbuf_offset = d.cbuf_offset;
    t.shift_left = d.shift_left, t.secondary_cbuf_index = d.secondary_cbuf_index;
    t.secondary_cbuf_offset = d.secondary_cbuf_offset, t.secondary_shift_left = d.secondary_shift_left;
    t.count = d.count, t.size_shift = d.size_shift;
    return t;
}
Shader::ImageDescriptor Clean(const Shader::ImageDescriptor& d) {
    auto t = Zeroed<Shader::ImageDescriptor>();
    t.type = d.type, t.format = d.format, t.is_written = d.is_written, t.is_read = d.is_read;
    t.is_integer = d.is_integer, t.cbuf_index = d.cbuf_index, t.cbuf_offset = d.cbuf_offset, t.count = d.count;
    t.size_shift = d.size_shift;
    return t;
}
template <typename T>
const T& Clean(const T& v) { // everything else (SPIR-V words, our own records): as it is
    return v;
}

struct Out {
    std::vector<u8>& b;
    template <typename T>
    void Pod(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* p = reinterpret_cast<const u8*>(&v);
        b.insert(b.end(), p, p + sizeof v);
    }
    template <size_t N>
    void Bits(const std::bitset<N>& v) {
        for (size_t i = 0; i < N; i += 8) {
            u8 byte = 0;
            for (size_t k = 0; k < 8 && i + k < N; ++k) byte |= static_cast<u8>(v[i + k]) << k;
            Pod(byte);
        }
    }
    template <typename C>
    void Seq(const C& c) {
        Pod(static_cast<u32>(c.size()));
        for (const auto& e : c) Pod(Clean(e));
    }
    template <typename M>
    void Map(const M& m) {
        Pod(static_cast<u32>(m.size()));
        for (const auto& [k, v] : m) Pod(k), Pod(v);
    }
};

struct In {
    const u8* p;
    const u8* end;
    bool ok = true;
    template <typename T>
    void Pod(T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (static_cast<size_t>(end - p) < sizeof v) {
            ok = false;
            return;
        }
        std::memcpy(&v, p, sizeof v);
        p += sizeof v;
    }
    template <size_t N>
    void Bits(std::bitset<N>& v) {
        v.reset();
        for (size_t i = 0; i < N; i += 8) {
            u8 byte = 0;
            Pod(byte);
            for (size_t k = 0; k < 8 && i + k < N; ++k) v[i + k] = (byte >> k) & 1;
        }
    }
    template <typename C>
    void Seq(C& c) {
        u32 n = 0;
        Pod(n);
        c.clear();
        for (u32 i = 0; i < n && ok; ++i) {
            typename C::value_type e{};
            Pod(e);
            if (c.size() == c.max_size()) { // a static_vector's capacity: a damaged entry
                ok = false;
                return;
            }
            c.push_back(e);
        }
    }
    template <typename M>
    void Map(M& m) {
        u32 n = 0;
        Pod(n);
        m.clear();
        for (u32 i = 0; i < n && ok; ++i) {
            typename M::key_type k{};
            typename M::mapped_type v{};
            Pod(k), Pod(v);
            m.emplace(k, v);
        }
    }
};

/// Every field of drippu's Shader::Info (shader_info.h), in its order; `I` is Info or const Info.
template <typename A, typename I>
void InfoFields(A& a, I& i) {
    a.Pod(i.uses_workgroup_id), a.Pod(i.uses_local_invocation_id), a.Pod(i.uses_invocation_id), a.Pod(i.uses_invocation_info);
    a.Pod(i.uses_sample_id), a.Pod(i.uses_is_helper_invocation), a.Pod(i.uses_subgroup_invocation_id);
    a.Pod(i.uses_subgroup_shuffles), a.Pod(i.uses_patches);
    a.Pod(i.interpolation);
    a.Bits(i.loads.mask), a.Bits(i.stores.mask), a.Bits(i.passthrough.mask);
    a.Map(i.legacy_stores_mapping);
    a.Pod(i.loads_indexed_attributes);
    a.Pod(i.stores_frag_color), a.Pod(i.stores_sample_mask), a.Pod(i.stores_frag_depth);
    a.Pod(i.stores_tess_level_outer), a.Pod(i.stores_tess_level_inner);
    a.Pod(i.stores_indexed_attributes);
    a.Pod(i.stores_global_memory), a.Pod(i.uses_local_memory);
    a.Pod(i.uses_fp16), a.Pod(i.uses_fp64), a.Pod(i.uses_fp16_denorms_flush), a.Pod(i.uses_fp16_denorms_preserve);
    a.Pod(i.uses_fp32_denorms_flush), a.Pod(i.uses_fp32_denorms_preserve);
    a.Pod(i.uses_int8), a.Pod(i.uses_int16), a.Pod(i.uses_int64);
    a.Pod(i.uses_image_1d), a.Pod(i.uses_sampled_1d), a.Pod(i.uses_sparse_residency), a.Pod(i.uses_demote_to_helper_invocation);
    a.Pod(i.uses_subgroup_vote), a.Pod(i.uses_subgroup_mask), a.Pod(i.uses_fswzadd), a.Pod(i.uses_derivatives);
    a.Pod(i.uses_typeless_image_reads), a.Pod(i.uses_typeless_image_writes), a.Pod(i.uses_image_buffers);
    a.Pod(i.uses_shared_increment), a.Pod(i.uses_shared_decrement), a.Pod(i.uses_global_increment), a.Pod(i.uses_global_decrement);
    a.Pod(i.uses_atomic_f32_add), a.Pod(i.uses_atomic_f16x2_add), a.Pod(i.uses_atomic_f16x2_min), a.Pod(i.uses_atomic_f16x2_max);
    a.Pod(i.uses_atomic_f32x2_add), a.Pod(i.uses_atomic_f32x2_min), a.Pod(i.uses_atomic_f32x2_max);
    a.Pod(i.uses_atomic_s32_min), a.Pod(i.uses_atomic_s32_max), a.Pod(i.uses_int64_bit_atomics), a.Pod(i.uses_global_memory);
    a.Pod(i.uses_atomic_image_u32), a.Pod(i.uses_shadow_lod), a.Pod(i.uses_rescaling_uniform), a.Pod(i.uses_cbuf_indirect);
    a.Pod(i.uses_render_area);
    a.Pod(i.used_constant_buffer_types), a.Pod(i.used_storage_buffer_types), a.Pod(i.used_indirect_cbuf_types);
    a.Pod(i.constant_buffer_mask), a.Pod(i.constant_buffer_used_sizes), a.Pod(i.nvn_buffer_base), a.Bits(i.nvn_buffer_used);
    a.Pod(i.requires_layer_emulation), a.Pod(i.emulated_layer);
    a.Pod(i.used_clip_distances);
    a.Seq(i.constant_buffer_descriptors), a.Seq(i.storage_buffers_descriptors), a.Seq(i.texture_buffer_descriptors);
    a.Seq(i.image_buffer_descriptors), a.Seq(i.texture_descriptors), a.Seq(i.image_descriptors);
}

std::vector<u8> Encode(const StoredStage& s) {
    std::vector<u8> b;
    Out o{b};
    o.Pod(kStoreVersion);
    o.Seq(s.spirv);
    InfoFields(o, s.info);
    o.Seq(s.cbuf_reads), o.Seq(s.type_reads);
    return b;
}

std::optional<StoredStage> Decode(const std::vector<u8>& b) {
    In in{b.data(), b.data() + b.size()};
    u32 version = 0;
    in.Pod(version);
    StoredStage s;
    in.Seq(s.spirv);
    InfoFields(in, s.info);
    in.Seq(s.cbuf_reads), in.Seq(s.type_reads);
    if (!in.ok || version != kStoreVersion || in.p != in.end || s.spirv.empty()) return std::nullopt;
    return s;
}

u64 Fnv(const u8* p, size_t n, u64 h) {
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

struct StageStore {
    std::once_flag once;
    std::mutex mu; // everything below
    std::condition_variable loaded_cv;
    bool loaded = false;
    std::filesystem::path path;
    std::string tag;
    bool tag_ok = false;  // the file has this tag (else it is rewritten at the first StorePut)
    u64 end = 0;          // where the next record goes (after the last whole one)
    std::unordered_map<std::string, std::pair<u64, u32>> index; // key -> {payload offset, size}
    std::fstream file;
    u64 hits = 0, misses = 0;
};

StageStore& TheStore() {
    static StageStore s;
    return s;
}

bool ReadU32(std::istream& in, u32& v) { return static_cast<bool>(in.read(reinterpret_cast<char*>(&v), 4)); }
void WriteU32(std::ostream& out, u32 v) { out.write(reinterpret_cast<const char*>(&v), 4); }

void Load(StageStore& s) {
    const auto t0 = std::chrono::steady_clock::now();
    s.tag = StoreTag();
    const char* dir = std::getenv("RT_PIPELINE_CACHE");
    char name[48];
    std::snprintf(name, sizeof name, "%016llx.nvn.stages", gpu::ProgramId());
    s.path = std::filesystem::path(dir && *dir && std::string(dir) != "0" ? dir : "cache") / name;
    const char* what = "none yet";
    if (std::ifstream in{s.path, std::ios::binary}) {
        char magic[8]{};
        u32 tag_size = 0;
        std::string tag;
        if (in.read(magic, 8) && std::memcmp(magic, kStoreMagic, 8) == 0 && ReadU32(in, tag_size) && tag_size < 4096) {
            tag.resize(tag_size);
            in.read(tag.data(), tag_size);
        }
        if (in && tag == s.tag) {
            s.tag_ok = true;
            s.end = 12 + tag_size;
            // records: {key size, key, payload size, payload}; a record cut short (a killed seed) ends the file
            in.seekg(0, std::ios::end);
            const u64 file_size = static_cast<u64>(in.tellg());
            in.seekg(static_cast<std::streamoff>(s.end));
            for (;;) {
                u32 key_size = 0, size = 0;
                std::string key;
                if (s.end + 8 > file_size || !ReadU32(in, key_size) || key_size > 4096) break;
                key.resize(key_size);
                if (!in.read(key.data(), key_size) || !ReadU32(in, size)) break;
                const u64 at = s.end + 8 + key_size;
                if (at + size > file_size) break;
                in.seekg(static_cast<std::streamoff>(at + size));
                s.index.emplace(std::move(key), std::pair<u64, u32>{at, size});
                s.end = at + size;
            }
            what = "loaded";
        } else {
            what = "made for another device, driver or translator: ignored";
        }
    }
    if (s.tag_ok) s.file.open(s.path, std::ios::binary | std::ios::in | std::ios::out);
    Log("nvn: shader store %s: %s (%zu stages; %.0f ms, on a thread of its own)", s.path.string().c_str(), what, s.index.size(),
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

StageStore& Ready() {
    StageStore& s = TheStore();
    StartShaderStoreLoad();
    std::unique_lock lock(s.mu);
    s.loaded_cv.wait(lock, [&] { return s.loaded; });
    return s;
}

} // namespace

bool ShaderStoreOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_SHADER_STORE");
        return !(e && *e == '0') && RescalingPassOn(); // entries are made with the pass
    }();
    return on;
}

bool ShaderStoreCheck() {
    static const bool on = [] {
        const char* e = std::getenv("RT_SHADER_STORE_CHECK");
        return e && *e && *e != '0';
    }();
    return on;
}

std::string StoreKey(const std::vector<u8>& code, const ProgramStage& p, u32 stage, const std::string& assumptions) {
    const auto region = [&](u64 at, u64 n) { return std::pair{code.data() + std::min<u64>(at, code.size()), std::min<u64>(n, code.size() - std::min<u64>(at, code.size()))}; };
    const auto [prog, prog_n] = region(p.program_offset, p.program_size);
    const auto [cons, cons_n] = region(p.constants_offset, p.constants_size);
    // two different hashes over both regions (128 bits: ~46 k stages never collide by chance)
    const u64 h1 = Fnv(cons, cons_n, Fnv(prog, prog_n, 0xcbf29ce484222325ull));
    const u64 h2 = Fnv(prog, prog_n, Fnv(cons, cons_n, 0x84222325cbf29ce4ull) ^ 0x9e3779b97f4a7c15ull);
    std::string key;
    const auto put = [&](const auto& v) { key.append(reinterpret_cast<const char*>(&v), sizeof v); };
    put(h1), put(h2), put(stage), put(p.stage);
    put(p.program_offset), put(p.program_size), put(p.constants_offset), put(p.constants_size);
    put(p.local_memory), put(p.workgroup), put(p.shared_memory), put(p.tag_word);
    key += assumptions;
    return key;
}

void StartShaderStoreLoad() {
    if (!ShaderStoreOn()) return;
    StageStore& s = TheStore();
    std::call_once(s.once, [&s] {
        std::thread([&s] {
            {
                std::lock_guard lock(s.mu);
                Load(s);
                s.loaded = true;
            }
            s.loaded_cv.notify_all();
        }).detach();
    });
}

std::optional<StoredStage> StoreGet(const std::string& key) {
    if (!ShaderStoreOn()) return std::nullopt;
    StageStore& s = Ready();
    std::lock_guard lock(s.mu);
    const auto it = s.index.find(key);
    std::optional<StoredStage> out;
    if (it != s.index.end() && s.file.is_open()) {
        std::vector<u8> b(it->second.second);
        s.file.clear();
        s.file.seekg(static_cast<std::streamoff>(it->second.first));
        if (s.file.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()))) out = Decode(b);
        if (!out) Log("nvn: shader store: an entry could not be read (damaged?): translated instead");
    }
    const u64 n = ++(out ? s.hits : s.misses);
    if (n == 1 || n == 10 || n == 100 || n % 500 == 0) {
        Log("nvn: shader store: %llu stages found, %llu not (RT_SHADER_STORE)", (unsigned long long)s.hits, (unsigned long long)s.misses);
    }
    return out;
}

void StorePut(const std::string& key, const StoredStage& entry) {
    if (!ShaderStoreOn()) return;
    StageStore& s = Ready();
    const std::vector<u8> payload = Encode(entry);
    std::lock_guard lock(s.mu);
    if (s.index.contains(key)) return;
    if (!s.tag_ok) { // a new file (or one of another tag): rewritten from its header
        std::error_code ec;
        std::filesystem::create_directories(s.path.parent_path(), ec);
        s.file.close();
        s.file.open(s.path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!s.file) {
            static bool told = false;
            if (!told) Log("nvn: shader store: cannot write %s", s.path.string().c_str()), told = true;
            return;
        }
        s.file.write(kStoreMagic, 8);
        WriteU32(s.file, static_cast<u32>(s.tag.size()));
        s.file.write(s.tag.data(), static_cast<std::streamsize>(s.tag.size()));
        s.index.clear();
        s.end = 12 + s.tag.size();
        s.tag_ok = true;
    }
    s.file.clear();
    s.file.seekp(static_cast<std::streamoff>(s.end));
    WriteU32(s.file, static_cast<u32>(key.size()));
    s.file.write(key.data(), static_cast<std::streamsize>(key.size()));
    WriteU32(s.file, static_cast<u32>(payload.size()));
    s.file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (!s.file) {
        static bool told = false;
        if (!told) Log("nvn: shader store: writing %s failed", s.path.string().c_str()), told = true;
        return;
    }
    const u64 at = s.end + 8 + key.size();
    s.index.emplace(key, std::pair<u64, u32>{at, static_cast<u32>(payload.size())});
    s.end = at + payload.size();
}

void StoreFlush() {
    if (!ShaderStoreOn()) return;
    StageStore& s = Ready();
    std::lock_guard lock(s.mu);
    if (s.file.is_open()) s.file.flush();
    Log("nvn: shader store: %zu stages in %s (%.0f MB)", s.index.size(), s.path.string().c_str(), s.end / 1048576.0);
}

std::string StoreTag() {
    const VkPhysicalDeviceProperties props = Vk().device->GetPhysical().GetProperties();
    char uuid[2 * VK_UUID_SIZE + 1]{};
    for (u32 i = 0; i < VK_UUID_SIZE; ++i) std::snprintf(uuid + 2 * i, 3, "%02x", props.pipelineCacheUUID[i]);
    char tag[384];
    std::snprintf(tag, sizeof tag, "store %u; drippu %s; %s; driver 0x%x; cache %s", kStoreVersion, RT_DRIPPU_COMMIT,
                  props.deviceName, props.driverVersion, uuid);
    return tag;
}

bool StoreFilled() {
    if (!ShaderStoreOn()) return true; // (nothing to fill)
    StageStore& s = Ready();
    std::lock_guard lock(s.mu);
    return s.tag_ok && !s.index.empty();
}

std::vector<u8> InfoBytes(const Shader::Info& info) {
    std::vector<u8> b;
    Out o{b};
    InfoFields(o, info);
    return b;
}

namespace {
/// Out that also notes where each of InfoFields' fields ends (InfoDifference).
struct MarkedOut {
    std::vector<u8> b;
    std::vector<size_t> ends;
    Out o{b};
    template <typename T> void Pod(const T& v) { o.Pod(v), ends.push_back(b.size()); }
    template <typename T> void Bits(const T& v) { o.Bits(v), ends.push_back(b.size()); }
    template <typename T> void Seq(const T& v) { o.Seq(v), ends.push_back(b.size()); }
    template <typename T> void Map(const T& v) { o.Map(v), ends.push_back(b.size()); }
};
} // namespace

std::string InfoDifference(const Shader::Info& a, const Shader::Info& b) {
    MarkedOut x, y;
    InfoFields(x, a), InfoFields(y, b);
    for (size_t f = 0; f < x.ends.size() && f < y.ends.size(); ++f) {
        const size_t from = f ? x.ends[f - 1] : 0;
        if (x.ends[f] != y.ends[f] || !std::equal(x.b.begin() + from, x.b.begin() + x.ends[f], y.b.begin() + from)) {
            size_t at = from;
            while (at < x.ends[f] && at < y.ends[f] && x.b[at] == y.b[at]) ++at;
            char s[96];
            std::snprintf(s, sizeof s, "field %zu of InfoFields (byte %zu of it; %zu vs %zu bytes)", f, at - from,
                          x.ends[f] - from, y.ends[f] - from);
            return s;
        }
    }
    return "none";
}

} // namespace rt::nvn
