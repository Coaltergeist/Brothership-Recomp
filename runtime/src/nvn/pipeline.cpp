// Graphics pipelines (NVN_PLAN 7.4).
//
// A pipeline is made from the queue's RenderState at a draw: the bound program's stages (vertex, geometry,
// fragment; tessellation not yet) translated together by drippu's recompiler — one binding numbering for
// all stages (Bindings), each stage's runtime info from the previous stage's outputs as drippu's pipeline
// cache makes it — and the fixed-function state from NVN's state objects (state.h) with viewport, scissor,
// stencil values, blend constants and depth bounds dynamic. Descriptors are pushed (KHR_push_descriptor)
// through the update template of drippu's DescriptorLayoutBuilder, so draws fill one flat array in its
// order (draws.cpp); a pipeline with more than the device pushes (32 on NVIDIA; ~13k draws in 3 min of
// battle, likely the water and the battle enemies) gets the same template for a descriptor set, one per draw from the
// submission's DescriptorArena (as ssbu-recomp does). RT_NVN_PUSH=0: sets everywhere.
//
// Cache: by the programs (code address and control fields), vertex layout, state objects, primitive and
// target formats; within one key, by what the translation read from the bound state (constant-buffer
// words, texture types), checked again at each lookup.
//
// On disk: the driver's VkPipelineCache, one file per game (<RT_PIPELINE_CACHE or "cache">/<program id>
// .nvn.vkpc), passed to every pipeline creation, the seed's included (2026-10-24: NVIDIA's own cache holds
// ~1 GB in all and dropped the seed's compute pipelines; LASRecomp and ssbu-recomp keep everything in a file
// of their own, ssbu-recomp's up to 0.9 GB). Read on a thread of its own as soon as the device exists
// (StartPipelineCacheLoad), used when its header names this device and driver (vendor, device,
// pipelineCacheUUID), else started empty; the first pipeline waits for the load if it is not done. Written
// (into a temporary file renamed over the old one, by a helper thread) only at safe points: reading a big cache
// out takes the driver's lock and holds every pipeline creation meanwhile (ssbu-recomp: freezes up to a minute
// with 0.9 GB) - at Exit (SavePipelineCacheAtExit: the window closed, the guest exited; a kill or crash runs
// nothing) when at least 8 pipelines were added or a seed is to be kept, and at a quiet moment in play (at least
// 16 added, none created for 60 s, 5 min since the last write; SavePipelineCacheIfChanged at each present).
// RT_PIPELINE_CACHE=0: no cache. It saves the driver's compilation, not drippu's translation.
// Each new pipeline logs one line (BuildOrRemember, SHADER_PLAN Phase 0): its total time, split into
// translation and driver, whether the driver found it in the pipeline cache (Vulkan 1.3 creation feedback:
// "cache hit" / "cache miss"; "cache ?" without it) and its stages' code hashes (StageHash: content, not
// address); the cache's write lines and Exit add the run's totals (PipelineTotals).
//
// NVN enum values (NVN's headers, as LASRecomp's runtime/nvn/vulkan/exec.c uses them, MIT): blend factors
// 1-11, 0x10-0x13, 0x61-0x64; blend equations 1-5; compare functions 1 never .. 8 always; stencil
// operations 1 keep .. 8 decrement-wrap; primitives in GL order; polygon state cull 0 none / 1 front /
// 2 back / 3 both, front face 0 CW / 1 CCW, mode 0 point / 1 line / 2 fill, offset enables 1 point /
// 2 line / 4 fill.
// drippu's headers first: vk.h brings X11's `Success` macro, which breaks its NvResult::Success.
#include "video_core/renderer_vulkan/pipeline_helper.h"

#include "pipeline.h"
#include "shader_store.h"
#include "shaders.h"
#include "state.h"
#include "../gpu.h"

#include <atomic>
#include <bit>
#include <string_view>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <thread>
#include <type_traits>
#include <string>
#include <map>
#include <set>
#include <span>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/runtime_info.h"

namespace rt::nvn {

// --- Pipeline cache on disk ---------------------------------------------------------------------

namespace {

struct Disk {
    std::once_flag once;                  // the load started (StartPipelineCacheLoad)
    std::mutex mu;                        // loaded, marker
    std::condition_variable loaded_cv;
    std::atomic<bool> loaded{false};      // set under `mu` (loaded_cv); read without it where a stale false is fine
    Vulkan::vk::PipelineCache cache;
    std::filesystem::path path;
    std::atomic<u64> made{0}, made_ns{0}; // pipelines created, and their creation time
    std::atomic<u64> written_made{0};     // `made` at the last write
    std::atomic<s64> last_made_ns{0};     // when the last pipeline was created (steady clock), for quiet moments
    std::atomic<bool> writing{false};
    std::chrono::steady_clock::time_point last_write{};
    std::filesystem::path marker;         // the seed's marker, written with the next write of the file (Seed)
    std::string marker_tag;
};

/// The run's pipeline builds (SHADER_PLAN Phase 0).
struct Totals {
    std::atomic<u64> graphics{0}, compute{0}, hits{0}, misses{0}, slow{0}, ns{0}, worst_ns{0};
    std::atomic<u64> stages{0}, stage_reuses{0}, stages_for_previous{0}; // RT_NVN_STAGE_ALONE (StageFor)
    std::atomic<u64> stages_stored{0}; // stages (and compute programs) taken from the shader store, not translated
    std::atomic<u64> not_ready{0}; // draws skipped while their pipeline was being built (PipelineFor)
};
Totals& TheTotals() {
    static Totals t;
    return t;
}
constexpr double kSlowBuildMs = 20;

std::string LibraryTotals();

/// "N pipelines built (G graphics, C compute): H driver cache hits, M misses; X ms in all, worst Y ms, Z over
/// 20 ms; S stages translated alone, R reused, P for their previous stage".
std::string PipelineTotals() {
    const Totals& t = TheTotals();
    char b[400];
    std::snprintf(b, sizeof b,
                  "%llu pipelines built (%llu graphics, %llu compute): %llu driver cache hits, %llu misses; %.0f ms in "
                  "all, worst %.1f ms, %llu over %.0f ms; %llu stages translated alone, %llu reused, %llu for their "
                  "previous stage, %llu from the shader store",
                  (unsigned long long)(t.graphics + t.compute), (unsigned long long)t.graphics.load(),
                  (unsigned long long)t.compute.load(), (unsigned long long)t.hits.load(),
                  (unsigned long long)t.misses.load(), t.ns / 1e6, t.worst_ns / 1e6, (unsigned long long)t.slow.load(),
                  kSlowBuildMs, (unsigned long long)t.stages.load(), (unsigned long long)t.stage_reuses.load(),
                  (unsigned long long)t.stages_for_previous.load(), (unsigned long long)t.stages_stored.load());
    return b + LibraryTotals() + "; " + std::to_string(t.not_ready.load()) + " draws skipped while their pipeline was built";
}

Disk& TheDisk() {
    static Disk d;
    return d;
}

/// The header of a pipeline cache (VkPipelineCacheHeaderVersionOne) names this device and driver.
bool HeaderMatches(const std::vector<char>& data, const VkPhysicalDeviceProperties& props) {
    if (data.size() < 32) return false;
    u32 words[4];
    std::memcpy(words, data.data(), 16);
    return words[0] >= 32 && words[1] == VK_PIPELINE_CACHE_HEADER_VERSION_ONE && words[2] == props.vendorID &&
           words[3] == props.deviceID && std::memcmp(data.data() + 16, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
}

/// Reads the file and gives it to the driver (StartPipelineCacheLoad's thread).
void LoadDiskCache(Disk& d) {
    const auto t0 = std::chrono::steady_clock::now();
    double read_ms = 0;
    [&] {
        const char* dir = std::getenv("RT_PIPELINE_CACHE");
        if (dir && std::string(dir) == "0") {
            Log("nvn: pipeline cache: off (RT_PIPELINE_CACHE=0)");
            return;
        }
        char name[48];
        std::snprintf(name, sizeof name, "%016llx.nvn.vkpc", gpu::ProgramId());
        d.path = std::filesystem::path(dir && *dir ? dir : "cache") / name;
        const auto& device = *Vk().device;
        const VkPhysicalDeviceProperties props = device.GetPhysical().GetProperties();
        std::vector<char> data;
        if (std::ifstream in{d.path, std::ios::binary | std::ios::ate}) {
            data.resize(static_cast<size_t>(in.tellg()));
            in.seekg(0);
            if (!in.read(data.data(), static_cast<std::streamsize>(data.size()))) data.clear();
        }
        read_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const char* what = "none yet";
        if (!data.empty() && !HeaderMatches(data, props)) {
            what = "made by another device or driver: ignored";
            data.clear();
        } else if (!data.empty()) {
            what = "loaded";
        }
        d.cache = device.GetLogical().CreatePipelineCache({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data.size(),
            .pInitialData = data.empty() ? nullptr : data.data(),
        });
        d.last_write = std::chrono::steady_clock::now();
        const double all_ms = std::chrono::duration<double, std::milli>(d.last_write - t0).count();
        Log("nvn: pipeline cache %s: %s (%zu KB; file read %.0f ms, the driver took it in %.0f ms, on a thread of its "
            "own)", d.path.string().c_str(), what, data.size() / 1024, read_ms, all_ms - read_ms);
    }();
    {
        std::lock_guard lock(d.mu);
        d.loaded = true;
    }
    d.loaded_cv.notify_all();
}

} // namespace

void StartPipelineCacheLoad() {
    Disk& d = TheDisk();
    std::call_once(d.once, [&d] { std::thread([&d] { LoadDiskCache(d); }).detach(); });
    StartShaderStoreLoad();
}

namespace {

/// The driver's pipeline cache for this run (VK_NULL_HANDLE with RT_PIPELINE_CACHE=0): waits for its load.
VkPipelineCache DiskCache() {
    Disk& d = TheDisk();
    StartPipelineCacheLoad();
    std::unique_lock lock(d.mu);
    if (!d.loaded) {
        const auto t0 = std::chrono::steady_clock::now();
        d.loaded_cv.wait(lock, [&] { return d.loaded.load(); });
        Log("nvn: a pipeline waited %.0f ms for the pipeline cache's load",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return d.cache ? *d.cache : VK_NULL_HANDLE;
}

thread_local u64 t_driver_ns = 0; // the last pipeline creation's driver time on this thread (BuildOrRemember)
thread_local int t_cache_hit = -1; // ... and whether the driver's cache had it (1 / 0; -1: not reported)

/// Pipeline creation feedback (core in Vulkan 1.3): whether the driver found the pipeline in the cache.
bool FeedbackOn() {
    static const bool on = Vk().device->ApiVersion() >= VK_API_VERSION_1_3;
    return on;
}

/// Counts a pipeline creation and its time (the cache's effect shows in the time per pipeline); Next() goes
/// into the creation's pNext for the driver's feedback.
struct PipelineTimer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    VkPipelineCreationFeedback feedback{};
    VkPipelineCreationFeedbackCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
        .pNext = nullptr,
        .pPipelineCreationFeedback = &feedback,
        .pipelineStageCreationFeedbackCount = 0,
        .pPipelineStageCreationFeedbacks = nullptr,
    };
    PipelineTimer() { t_cache_hit = -1; }
    PipelineTimer(const PipelineTimer&) = delete;
    const void* Next() const { return FeedbackOn() ? &info : nullptr; }
    ~PipelineTimer() {
        Disk& d = TheDisk();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        t_driver_ns = static_cast<u64>(ns);
        if (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) {
            t_cache_hit = (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) ? 1 : 0;
        }
        d.made_ns.fetch_add(static_cast<u64>(ns), std::memory_order_relaxed);
        d.made.fetch_add(1, std::memory_order_relaxed);
        d.last_made_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);
    }
};

/// `why`: "at exit" / "at a quiet moment".
void WriteCache(Disk& d, const char* why) {
    const auto t0 = std::chrono::steady_clock::now();
    size_t size = 0;
    std::vector<char> data;
    if (d.cache.Read(&size, nullptr) == VK_SUCCESS && size) {
        data.resize(size);
        if (d.cache.Read(&size, data.data()) != VK_SUCCESS) size = 0;
    }
    const double read_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (size) {
        std::error_code ec;
        std::filesystem::create_directories(d.path.parent_path(), ec);
        auto tmp = d.path;
        tmp += ".tmp";
        bool ok = false;
        if (std::ofstream out{tmp, std::ios::binary | std::ios::trunc}) {
            out.write(data.data(), static_cast<std::streamsize>(size));
            ok = static_cast<bool>(out);
        }
        if (ok) std::filesystem::rename(tmp, d.path, ec);
        const u64 made = d.made.load(), ns = d.made_ns.load();
        Log("nvn: pipeline cache written %s (%zu KB%s; read out of the driver in %.0f ms, the file in %.0f ms); %llu "
            "pipelines made so far, %.1f ms on average; %s", why, size / 1024, ok && !ec ? "" : ", FAILED", read_ms,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() - read_ms,
            (unsigned long long)made, made ? ns / 1e6 / made : 0.0, PipelineTotals().c_str());
        std::lock_guard lock(d.mu);
        if (ok && !ec && !d.marker.empty()) { // the seed is in the file now: the next start skips it
            std::ofstream(d.marker) << d.marker_tag;
            Log("nvn seed: marker %s written with the pipeline cache", d.marker.string().c_str());
            d.marker.clear();
        }
    }
    d.writing.store(false);
}

} // namespace

void SavePipelineCacheIfChanged() {
    Disk& d = TheDisk();
    if (!d.loaded || !d.cache) return; // (a stale `loaded` only skips one present)
    const u64 made = d.made.load(std::memory_order_relaxed);
    if (made < d.written_made.load(std::memory_order_relaxed) + 16) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - d.last_write < std::chrono::minutes(5)) return;
    const s64 last = d.last_made_ns.load(std::memory_order_relaxed);
    if (now.time_since_epoch().count() - last < s64{60} * 1'000'000'000) return; // a pipeline in the last minute
    if (d.writing.exchange(true)) return;
    d.last_write = now;
    d.written_made.store(made);
    std::thread([&d] { // (VkPipelineCache is synchronised by the driver)
        WriteCache(d, "at a quiet moment");
    }).detach();
}

void SavePipelineCacheAtExit() {
    Disk& d = TheDisk();
    for (int i = 0; i < 1000 && d.writing.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Log("nvn: at exit: %s", PipelineTotals().c_str());
    if (!d.loaded || !d.cache) return;
    const u64 added = d.made.load() - d.written_made.load();
    bool marker_due;
    {
        std::lock_guard lock(d.mu);
        marker_due = !d.marker.empty();
    }
    if (added < 8 && !marker_due) { // rewriting a big file for a few pipelines is not worth it: made again next time
        if (added) Log("nvn: pipeline cache not written at exit: %llu pipelines added (under 8)", (unsigned long long)added);
        return;
    }
    if (d.writing.exchange(true)) return;
    d.written_made.store(d.made.load());
    WriteCache(d, "at exit");
}

namespace {

constexpr u32 kVertex = 0, kFragment = 1, kGeometry = 2, kTessControl = 3, kTessEvaluation = 4;
constexpr u32 kPipelineOrder[] = {kVertex, kTessControl, kTessEvaluation, kGeometry, kFragment};

VkShaderStageFlagBits VkStage(u32 s) {
    switch (s) {
    case kVertex: return VK_SHADER_STAGE_VERTEX_BIT;
    case kFragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
    case kGeometry: return VK_SHADER_STAGE_GEOMETRY_BIT;
    case kTessControl: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case kTessEvaluation: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    }
    Fatal("nvn: graphics stage %u", s);
}

[[noreturn]] void Unsupported(const char* fmt, auto... args) {
    char b[256];
    std::snprintf(b, sizeof b, fmt, args...);
    throw NotYet(b);
}

/// RT_NVN_DUMP_SHADERS=<dir>: a translated stage's instructions (<hash>.code, from the program offset) and
/// drippu's translation (<hash>.ir.txt), by the "new pipeline" lines' hash, for reading a shader by hand.
std::vector<u8> ReadCode(const ProgramStage& s);
u64 StageHash(const ProgramStage& s);

void DumpStage(const ProgramStage& s, const Shader::IR::Program& ir) {
    const char* dir = std::getenv("RT_NVN_DUMP_SHADERS");
    if (!dir) return;
    char name[32];
    std::snprintf(name, sizeof name, "%016llx", (unsigned long long)StageHash(s));
    std::filesystem::create_directories(dir);
    const std::vector<u8> code = ReadCode(s);
    const u64 begin = std::min<u64>(s.program_offset, code.size());
    const u64 end = std::min<u64>(begin + s.program_size, code.size());
    std::ofstream(std::filesystem::path(dir) / (std::string(name) + ".code"), std::ios::binary)
        .write(reinterpret_cast<const char*>(code.data() + begin), static_cast<std::streamsize>(end - begin));
    std::ofstream(std::filesystem::path(dir) / (std::string(name) + ".ir.txt")) << Shader::IR::DumpProgram(ir);
}

/// Decorates the Position output of a translated stage Invariant (SPIR-V, patched after drippu's emitter): the
/// game's depth prepass and base pass draw the same mesh with different vertex shaders and test depth for
/// equality (GEQUAL, depth writes off), which the Switch's compiler keeps exact; without the decoration the
/// host driver may round the two positions differently and drop or expose whole triangles (2026-10-21: the
/// morphed faces on the results screen). DXVK decorates every vertex shader so for the same reason.
/// RT_NVN_INVARIANT=0: off (A/B).
void MakePositionInvariant(std::vector<u32>& spirv) {
    static const bool off = [] {
        const char* e = std::getenv("RT_NVN_INVARIANT");
        return e && std::strcmp(e, "0") == 0;
    }();
    if (off) return;
    constexpr u32 kOpVariable = 59, kOpDecorate = 71, kBuiltIn = 11, kPosition = 0, kInvariant = 18, kOutput = 3;
    std::map<u32, size_t> position; // decorated id -> the word after its BuiltIn Position decoration
    std::set<u32> outputs;
    for (size_t i = 5; i < spirv.size();) {
        const u32 words = spirv[i] >> 16, op = spirv[i] & 0xFFFF;
        if (words == 0 || i + words > spirv.size()) return; // not what drippu emits: leave it
        if (op == kOpDecorate && words >= 4 && spirv[i + 2] == kBuiltIn && spirv[i + 3] == kPosition) position[spirv[i + 1]] = i + words;
        if (op == kOpVariable && words >= 4 && spirv[i + 3] == kOutput) outputs.insert(spirv[i + 2]);
        i += words;
    }
    for (const auto& [id, at] : position) {
        if (!outputs.count(id)) continue;
        spirv.insert(spirv.begin() + static_cast<std::ptrdiff_t>(at), {(3u << 16) | kOpDecorate, id, kInvariant});
        return; // one Position output per stage
    }
}

/// Whether a pipeline with `builder`'s descriptors pushes them; else a descriptor set per draw (DescriptorArena).
/// RT_NVN_PUSH=0: sets for every pipeline (a test of that path where nothing goes over the limit).
bool UsePush(const Vulkan::DescriptorLayoutBuilder& builder) {
    static const bool never = [] {
        const char* e = std::getenv("RT_NVN_PUSH");
        return e && std::strcmp(e, "0") == 0;
    }();
    return !never && builder.CanUsePushDescriptor();
}

VkBlendFactor BlendFactor(u8 f) {
    switch (f) {
    case 1: return VK_BLEND_FACTOR_ZERO;
    case 2: return VK_BLEND_FACTOR_ONE;
    case 3: return VK_BLEND_FACTOR_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    }
    Unsupported("blend factor %u", f);
}

VkBlendOp BlendOp(u8 e) {
    if (e < 1 || e > 5) Unsupported("blend equation %u", e);
    constexpr VkBlendOp ops[] = {VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_REVERSE_SUBTRACT, VK_BLEND_OP_MIN,
                                 VK_BLEND_OP_MAX};
    return ops[e - 1];
}

VkCompareOp CompareOp(u32 f) {
    if (f < 1 || f > 8) Unsupported("compare function %u", f);
    return static_cast<VkCompareOp>(f - 1); // NVN 1 never .. 8 always = Vulkan 0 .. 7
}

VkStencilOp StencilOp(u32 o) {
    if (o < 1 || o > 8) Unsupported("stencil operation %u", o);
    return static_cast<VkStencilOp>(o - 1); // NVN 1 keep .. 8 decrement-wrap = Vulkan 0 .. 7
}

VkStencilOpState StencilState(u8 func, u16 ops, const RenderState& rs, u32 face) {
    return {StencilOp(ops & 15), StencilOp((ops >> 8) & 15), StencilOp((ops >> 4) & 15), CompareOp(func),
            rs.stencil_value_mask[face], rs.stencil_mask[face], rs.stencil_ref[face]};
}

VkPrimitiveTopology Topology(u32 mode) {
    switch (mode) {
    case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case 4: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case 6: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case 10: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    case 11: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case 12: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    case 13: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    }
    Unsupported("primitive mode %u (line loops, quads, polygons, patches)", mode);
}

Shader::InputTopology GeometryInput(u32 mode) {
    switch (mode) {
    case 0: return Shader::InputTopology::Points;
    case 1:
    case 3: return Shader::InputTopology::Lines;
    case 10:
    case 11: return Shader::InputTopology::LinesAdjacency;
    case 12:
    case 13: return Shader::InputTopology::TrianglesAdjacency;
    default: return Shader::InputTopology::Triangles;
    }
}

/// Integer formats are read and written as integers by shaders.
Shader::AttributeType FormatType(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8_UINT: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16_UINT:
    case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_S8_UINT:
        return Shader::AttributeType::UnsignedInt;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8_SINT: case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16_SINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32A32_SINT:
        return Shader::AttributeType::SignedInt;
    default: return Shader::AttributeType::Float;
    }
}

/// A vertex format the device can read: 3-component 8/16-bit formats are read as 4 components where it
/// cannot (LASRecomp's fallback; the fourth component is the next bytes of the stream).
VkFormat SupportedVertexFormat(VkFormat f) {
    static std::mutex mu;
    static std::map<VkFormat, bool> ok;
    std::lock_guard lock(mu);
    auto it = ok.find(f);
    if (it == ok.end()) {
        const auto props = Vk().device->GetPhysical().GetFormatProperties(f);
        it = ok.emplace(f, (props.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) != 0).first;
    }
    if (it->second) return f;
    switch (f) {
    case VK_FORMAT_R8G8B8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case VK_FORMAT_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VK_FORMAT_R16G16B16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case VK_FORMAT_R16G16B16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    default: Unsupported("vertex format %d", static_cast<int>(f));
    }
}

std::vector<u8> ReadCode(const ProgramStage& s);

/// FNV-1a (64-bit) of a stage's instructions and compiler constants, read from its code memory: what the
/// stage is, whatever its address (SHADER_PLAN Phase 1 keys by it; the debug tag at +0 is left out).
/// StageHash of `s`'s code memory `code` (as ReadCode reads it).
u64 CodeHash(const std::vector<u8>& code, const ProgramStage& s) {
    u64 h = 0xcbf29ce484222325ull;
    const auto add = [&](u64 at, u64 n) {
        for (u64 i = at; i < at + n && i < code.size(); ++i) h = (h ^ code[i]) * 0x100000001b3ull;
    };
    add(s.program_offset, s.program_size);
    add(s.constants_offset, s.constants_size);
    return h;
}

u64 StageHash(const ProgramStage& s) { return CodeHash(ReadCode(s), s); }

/// Appends the bytes of `v` to a cache key.
template <typename T>
void Put(std::vector<u8>& key, const T& v) {
    const auto* p = reinterpret_cast<const u8*>(&v);
    key.insert(key.end(), p, p + sizeof v);
}

/// One stage on its way through drippu's recompiler: the IR program lives in these pools until it is
/// emitted.
struct StageWork {
    u32 nvn_stage;
    ProgramStage program;
    std::unique_ptr<NvnEnvironment> env;
    std::unique_ptr<Shader::ObjectPool<Shader::Maxwell::Flow::Block>> flow_blocks;
    std::unique_ptr<Shader::ObjectPool<Shader::IR::Inst>> insts;
    std::unique_ptr<Shader::ObjectPool<Shader::IR::Block>> blocks;
    Shader::IR::Program ir;
    std::shared_ptr<bool> building_cfg; // ReadsThatPin
};

/// The stage's code memory from its GPU address: the debug tag, the SPH or instructions, the compiler
/// constants (control +0x704 = round256(constants offset + size) bytes).
std::vector<u8> ReadCode(const ProgramStage& s) {
    const u64 end = std::max<u64>(u64{s.program_offset} + s.program_size, u64{s.constants_offset} + s.constants_size);
    std::vector<u8> code((end + 255) & ~u64{255});
    g_mem.Read(s.code, code.data(), code.size());
    return code;
}

struct Cache {
    std::mutex mu;
    std::condition_variable built; // a worker finished a pipeline (PipelineFor)
    std::map<std::vector<u8>, std::vector<std::unique_ptr<GraphicsPipeline>>> pipelines;
};
Cache& Pipelines() {
    static Cache c;
    return c;
}

/// Logs a program whose pipeline keeps being rebuilt for new values of what its translation read (at 4, 8,
/// 16 ... variants), with the reads that pinned the newest one; and variants 2-8 with the texture types the newest
/// and the first read (handle: type), to see why a variant was needed (2026-10-24: 328 compute pipelines for 183
/// programs in the new-game run, 85 of them cold compiles).
template <typename P>
void NoteVariants(const char* kind, const std::vector<std::unique_ptr<P>>& variants) {
    const size_t n = variants.size();
    if (n >= 2 && n <= 8) {
        const auto types = [](const P& v) {
            std::string t;
            for (const auto& r : v.type_reads) {
                char b[48];
                std::snprintf(b, sizeof b, " %x:%u", r.handle, static_cast<u32>(r.type));
                t += b;
            }
            return t.empty() ? std::string(" none") : t;
        };
        Log("nvn %s: variant %zu of one program: texture types (handle:type) read%s; the first variant's%s", kind, n,
            types(*variants.back()).c_str(), types(*variants.front()).c_str());
    }
    const P& p = *variants.back();
    if (n < 4 || (n & (n - 1))) return;
    std::string reads;
    for (const auto& r : p.cbuf_reads) {
        char b[64];
        std::snprintf(b, sizeof b, " c%u[0x%x]=0x%x (stage %u)", r.bank, r.offset, r.value, r.nvn_stage);
        reads += b;
    }
    Log("nvn %s: %zu pipeline variants of one program; the newest read%s%s", kind, n, reads.empty() ? " nothing" : ":",
        reads.c_str());
}

/// Builds a new variant into `variants`. A build that fails (NotYet) is kept too, as a failed variant with
/// what its translation read: the same state fails the same way, so later draws skip it at lookup cost
/// instead of translating its shaders again (2026-10-19: ~10k such draws per minute of gameplay, 32 us of
/// every draw's 50 us on average).
template <typename P, typename F>
bool BuildLogged(P& fresh, const char* kind, F&& build);

template <typename P, typename F>
void BuildOrRemember(std::vector<std::unique_ptr<P>>& variants, const char* kind, F&& build) {
    auto fresh = std::make_unique<P>();
    const bool built = BuildLogged(*fresh, kind, std::forward<F>(build));
    variants.push_back(std::move(fresh));
    if (!built) throw NotYet(variants.back()->failed);
    NoteVariants(kind, variants);
}

/// Builds `fresh` with `build` and logs it; false when the build failed (NotYet: the reason in `fresh.failed`).
template <typename P, typename F>
bool BuildLogged(P& fresh_ref, const char* kind, F&& build) {
    P* const fresh = &fresh_ref;
    const auto t0 = std::chrono::steady_clock::now();
    t_driver_ns = 0;
    t_cache_hit = -1;
    try {
        build(*fresh);
    } catch (const NotYet& e) {
        fresh->failed = e.what();
        return false;
    } catch (const std::exception& e) {
        // drippu's translator gave up on a stage (2026-10-24: a compute shader's bindless texture it could not
        // track, in a new-game cutscene; it crashed the run): a failed variant like NotYet's, so the draw or
        // dispatch is skipped, logged once with the stage hashes (ReadCode's bytes saved for the shader test)
        std::string hashes;
        const auto add = [&](const ProgramStage& st) {
            char b[24];
            std::snprintf(b, sizeof b, "%s%016llx", hashes.empty() ? "" : "/", (unsigned long long)StageHash(st));
            hashes += b;
            const std::vector<u8> code = ReadCode(st);
            std::filesystem::create_directories("failed_shaders");
            std::ofstream(std::filesystem::path("failed_shaders") / (std::string(b + (b[0] == '/')) + ".code"),
                          std::ios::binary)
                .write(reinterpret_cast<const char*>(code.data()), static_cast<std::streamsize>(code.size()));
        };
        if constexpr (std::is_same_v<P, ComputePipeline>) add(fresh->program);
        else for (const auto& st : fresh->stages) add(st.program);
        fresh->failed = std::string("drippu's translator: ") + e.what();
        Log("nvn: %s pipeline failed to translate (t=%.2f s): %s; code %s (saved to failed_shaders/); its %s "
            "skipped", kind, Uptime(), e.what(), hashes.empty() ? "?" : hashes.c_str(),
            std::is_same_v<P, ComputePipeline> ? "dispatches are" : "draws are");
        return false;
    }
    // One line per new pipeline: when, its time split into drippu's translation (with layouts and modules)
    // and the driver's creation, whether the driver's cache had it, and its stages' code hashes - what a
    // hitch at a new pipeline is made of.
    const u64 ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    const double total = ns / 1e6, driver = t_driver_ns / 1e6;
    std::string hashes;
    char b[48];
    u32 descriptors = 0;
    if constexpr (std::is_same_v<P, ComputePipeline>) {
        std::snprintf(b, sizeof b, "%016llx", (unsigned long long)StageHash(fresh->program));
        hashes = b;
        descriptors = Vulkan::NumDescriptorEntries(fresh->info);
        // the texture types it was built for (handle:type, drippu's numbers; "?" a handle with no registered
        // texture, answered 2D; @bank:offset[|bank:offset] where the handle came from): to compare with the seed
        if (!fresh->type_reads.empty()) hashes += "; types";
        for (const auto& r : fresh->type_reads) {
            char t[64];
            std::snprintf(t, sizeof t, " %x:%u%s", r.handle, static_cast<u32>(r.type), r.registered ? "" : "?");
            hashes += t;
            if (r.words) {
                std::snprintf(t, sizeof t, "@%u:%x", r.bank, r.offset);
                hashes += t;
                if (r.words == 2) {
                    std::snprintf(t, sizeof t, "|%u:%x", r.bank2, r.offset2);
                    hashes += t;
                }
            }
        }
    } else {
        for (const auto& st : fresh->stages) {
            std::snprintf(b, sizeof b, "%s%016llx", hashes.empty() ? "" : "/", (unsigned long long)StageHash(st.program));
            hashes += b;
            descriptors += Vulkan::NumDescriptorEntries(st.info);
        }
    }
    if (!fresh->push) {
        std::snprintf(b, sizeof b, "; %u descriptors, a set per draw", descriptors);
        hashes += b;
    }
    Totals& t = TheTotals();
    const u64 n = (std::is_same_v<P, ComputePipeline> ? t.compute : t.graphics).fetch_add(1) + 1;
    if (t_cache_hit >= 0) (t_cache_hit ? t.hits : t.misses).fetch_add(1);
    t.ns.fetch_add(ns);
    for (u64 w = t.worst_ns.load(); ns > w && !t.worst_ns.compare_exchange_weak(w, ns);) {}
    if (total > kSlowBuildMs) {
        t.slow.fetch_add(1);
    }
    Log("nvn: new %s pipeline #%llu (t=%.2f s): %.1f ms (translation %.1f ms, driver %.1f ms, cache %s); code %s", kind,
        (unsigned long long)n, Uptime(), total, total - driver, driver,
        t_cache_hit == 1 ? "hit" : t_cache_hit == 0 ? "miss" : "?", hashes.c_str());
    return true;
}

// ReadsThatPin: which constant-buffer words a pipeline is tied to. drippu reads them in two places: while
// building the control-flow graph (indirect branch tables: the values become code) and in its texture pass,
// where a handle word is read only to ask the texture's type (kept in type_reads), pixel format and integer-
// ness (both fixed in NvnEnvironment). Only the first kind pins the pipeline; pinning handle values made a
// new variant for every new texture id the game bound (64+ compute variants on the title, one compile per
// frame).
/// Where a translation's texture handles came from. drippu reads a handle's word(s) through the cbuf hook (each
/// word once per environment: it caches them) and then asks the handle's type; Word() keeps those reads, Type()
/// answers the type and records the read with the word or the pair of words (OR-ed) that make the handle, when
/// exactly one does. The pipeline is then reused while the handle read *now* from those words has the same type
/// (TypeHolds). Checked by the old handle value instead (before 2026-10-24, and still when no single source is
/// found), a pipeline was rebuilt whenever the game reused the old handle's pool slot for another texture: in the
/// plaza 5 variants of one compute program, all reading the same types.
struct TypeRecorder {
    struct Read {
        u32 bank, offset, value;
    };
    std::vector<Read> reads;
    void Word(u32 bank, u32 offset, u32 value) {
        for (const auto& r : reads) {
            if (r.bank == bank && r.offset == offset) return;
        }
        reads.push_back({bank, offset, value});
    }
    GraphicsPipeline::TypeRead Type(const RenderState& rs, u32 stage, u32 handle) const {
        bool registered = true;
        GraphicsPipeline::TypeRead t{stage, handle, TextureTypeOf(rs, handle, &registered)};
        t.registered = registered;
        Locate(t);
        return t;
    }
    /// Where `t.handle` came from: the constant word (or two OR-ed words) read with its value (t.words 0: none or
    /// ambiguous, checked by the handle value).
    void Locate(GraphicsPipeline::TypeRead& t) const {
        const u32 handle = t.handle;
        int found = 0;
        for (const auto& r : reads) {
            if (r.value != handle) continue;
            if (++found == 1) t.words = 1, t.bank = r.bank, t.offset = r.offset;
        }
        if (found == 0) {
            for (size_t i = 0; i < reads.size(); ++i) {
                for (size_t j = i + 1; j < reads.size(); ++j) {
                    const Read &a = reads[i], &b = reads[j];
                    if (!a.value || !b.value || (a.value | b.value) != handle) continue;
                    if (++found == 1) t.words = 2, t.bank = a.bank, t.offset = a.offset, t.bank2 = b.bank, t.offset2 = b.offset;
                }
            }
        }
        if (found != 1) t.words = 0; // none or ambiguous: checked by the handle value
    }
};

/// Whether texture read `r` still gets the same type in state `rs` (TypeRecorder).
bool TypeHolds(const GraphicsPipeline::TypeRead& r, const RenderState& rs) {
    u32 handle = r.handle;
    if (r.words) {
        handle = CbufWord(rs, r.nvn_stage, r.bank, r.offset);
        if (r.words == 2) handle |= CbufWord(rs, r.nvn_stage, r.bank2, r.offset2);
    }
    return TextureTypeOf(rs, handle) == r.type;
}

template <typename P>
bool StillHolds(const P& p, const RenderState& rs) {
    for (const auto& r : p.cbuf_reads) {
        if (CbufWord(rs, r.nvn_stage, r.bank, r.offset) != r.value) return false;
    }
    for (const auto& r : p.type_reads) {
        if (!TypeHolds(r, rs)) return false;
    }
    return true;
}

// RT_NVN_STAGE_DEPS=1 (SHADER_PLAN: the measurements before choosing a route): for each stage of a new
// graphics pipeline, whether its SPIR-V depends on the pipeline's state, i.e. whether a stage built on its own
// (from the game files, before any draw) could serve this pipeline. The stage is translated again, each time
// with its own binding numbers (sharing them across stages is our choice, not state): with this pipeline's
// state, and "alone" - every generic input, float vertex attributes and colour outputs, no previous stage's
// legacy mapping, every texture 2D (nothing bound to ask); depth mode and geometry topology as here (the
// device's and the draw's). When the two differ, up to three more translations name the cause - the one of
// attribute/colour types, inputs (previous stage) or texture types that alone makes them equal, else
// "several". Pins (constant words read while building the control flow) are counted from the build itself.
bool StageAloneOn();

// RT_NVN_GPL (SHADER_PLAN Phase 2; default on where the device has it; =0: whole pipelines): graphics pipelines
// are linked from pipeline libraries (Build, LibraryFor) - one per stage build and render pass, so a new pipeline
// of known stages costs a link instead of a compile. Needs the stages built alone and the state the shader
// libraries would otherwise hold set at each draw (extended dynamic state 1 and 2).
bool GplOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_NVN_GPL");
        if (e && std::strcmp(e, "0") == 0) {
            Log("nvn: RT_NVN_GPL=0: whole graphics pipelines");
            return false;
        }
        const auto& d = *Vk().device;
        const char* missing = !StageAloneOn()                            ? "stages built alone (RT_NVN_STAGE_ALONE=0)"
                              : !d.IsGraphicsPipelineLibrarySupported()   ? "graphics pipeline libraries with fast linking"
                              : !d.IsExtExtendedDynamicStateSupported()   ? "extended dynamic state"
                              : !d.IsExtExtendedDynamicState2Supported()  ? "extended dynamic state 2"
                                                                          : nullptr;
        if (missing) {
            Log("nvn: whole graphics pipelines: no %s", missing);
            return false;
        }
        Log("nvn: graphics pipelines linked from pipeline libraries (RT_NVN_GPL)");
        return true;
    }();
    return on;
}

bool StageDepsOn() {
    static const bool on = std::getenv("RT_NVN_STAGE_DEPS") != nullptr;
    return on;
}

std::vector<u32> EmitWith(const ProgramStage& st, const RenderState& rs, u32 s, const Shader::RuntimeInfo& rt,
                          bool bound_textures) {
    ShaderHooks hooks{
        .cbuf = [&rs, s](u32 bank, u32 offset) { return CbufWord(rs, s, bank, offset); },
        .texture_type = [&rs, bound_textures](u32 handle) {
            return bound_textures ? TextureTypeOf(rs, handle) : Shader::TextureType::Color2D;
        },
    };
    NvnEnvironment env(ReadCode(st), st, std::move(hooks));
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
    Shader::ObjectPool<Shader::IR::Inst> insts;
    Shader::ObjectPool<Shader::IR::Block> blocks;
    Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, env.StartAddress() + static_cast<u32>(sizeof(Shader::ProgramHeader)), false);
    Shader::IR::Program ir = Shader::Maxwell::TranslateProgram(insts, blocks, env, cfg, ShaderHostInfo());
    RescalingPass(ir);
    Shader::RuntimeInfo copy = rt;
    Shader::Maxwell::ConvertLegacyToGeneric(ir, copy);
    Shader::Backend::Bindings bindings;
    SetRescaleBase(bindings, ir.stage);
    return Shader::Backend::SPIRV::EmitSPIRV(ShaderProfile(), copy, ir, bindings);
}

void LogStageDeps(const GraphicsPipeline& p, const ProgramStage& program, u32 s, const RenderState& rs,
                  const Shader::RuntimeInfo& rt) {
    Shader::RuntimeInfo alone;
    alone.previous_stage_stores.mask.set();
    alone.convert_depth_mode = rt.convert_depth_mode;
    alone.input_topology = rt.input_topology;
    const std::vector<u32> actual = EmitWith(program, rs, s, rt, true);
    const bool same = EmitWith(program, rs, s, alone, false) == actual;
    const char* cause = "";
    if (!same) {
        Shader::RuntimeInfo types = alone;
        types.generic_input_types = rt.generic_input_types;
        types.color_output_types = rt.color_output_types;
        Shader::RuntimeInfo inputs = alone;
        inputs.previous_stage_stores = rt.previous_stage_stores;
        inputs.previous_stage_legacy_stores_mapping = rt.previous_stage_legacy_stores_mapping;
        cause = EmitWith(program, rs, s, types, false) == actual    ? "types"
                : EmitWith(program, rs, s, inputs, false) == actual ? "inputs"
                : EmitWith(program, rs, s, alone, true) == actual   ? "textures"
                                                                      : "several";
    }
    u32 attribs = 0, colours = 0, pins = 0, types = 0;
    for (auto t : rt.generic_input_types) attribs += t != Shader::AttributeType::Float;
    for (auto t : rt.color_output_types) colours += t != Shader::AttributeType::Float;
    for (const auto& r : p.cbuf_reads) pins += r.nvn_stage == s;
    for (const auto& r : p.type_reads) types += r.nvn_stage == s && r.type != Shader::TextureType::Color2D;
    // what the state answered: the non-float attribute types (index:type) and every texture type asked
    // (handle:type), as drippu's AttributeType and TextureType numbers
    std::string answers;
    char b[40];
    for (u32 i = 0; i < rt.generic_input_types.size(); ++i) {
        if (rt.generic_input_types[i] == Shader::AttributeType::Float) continue;
        std::snprintf(b, sizeof b, " a%u:%u", i, static_cast<u32>(rt.generic_input_types[i]));
        answers += b;
    }
    for (const auto& r : p.type_reads) {
        if (r.nvn_stage != s) continue;
        std::snprintf(b, sizeof b, " t%x:%u", r.handle, static_cast<u32>(r.type));
        answers += b;
    }
    Log("nvn stagedeps: %016llx stage %u %s%s%s; non-float attributes %u, non-float colours %u, pins %u, non-2D textures %u;%s",
        (unsigned long long)StageHash(program), s, same ? "same" : "differs", same ? "" : " by ", cause, attribs,
        colours, pins, types, answers.c_str());
}

// RT_NVN_STAGE_ALONE (default on; =0: the per-pipeline translation in Build) - SHADER_PLAN Phase 1: every stage
// is translated on its own and shared by all pipelines that use it. Each has its own descriptor set (fragment 0,
// pushed when it fits; vertex 1; geometry 2 - drippu decorates everything with set 0, MoveDescriptorSet rewrites
// it) with its own binding numbers, and takes every input of a previous stage as present. What a stage still
// assumes is its key: its code memory (program and compiler constants), plus the attribute types (vertex), the
// colour output types (fragment), the depth mode and the geometry input topology; under one key, variants by the
// constant words read while building the control flow and the texture types asked (as pipelines' StillHolds).
// A fragment stage that reads a generic input its previous stage does not write, or a legacy input (mapped by the
// previous stage), is translated for that previous stage instead: keyed by its outputs too.
bool StageAloneOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_NVN_STAGE_ALONE");
        const bool alone = !(e && std::strcmp(e, "0") == 0);
        if (!alone) Log("nvn: RT_NVN_STAGE_ALONE=0: each pipeline translates its stages itself");
        return alone;
    }();
    return on;
}

constexpr u32 SetOf(u32 nvn_stage) { return nvn_stage == kFragment ? 0 : nvn_stage == kVertex ? 1 : 2; }

bool GplOn();

/// A graphics pipeline layout of `sets` (VK_NULL_HANDLE: a set this layout leaves to others, libraries only) and
/// drippu's push constants (as DescriptorLayoutBuilder::CreatePipelineLayout); with independent sets when
/// pipelines are linked from libraries, whose layouts each hold only their own stages' sets.
Vulkan::vk::PipelineLayout PipelineLayout(std::span<const VkDescriptorSetLayout> sets) {
    using Shader::Backend::SPIRV::RenderAreaLayout;
    using Shader::Backend::SPIRV::RescalingLayout;
    const VkPushConstantRange range{VK_SHADER_STAGE_ALL_GRAPHICS, 0,
                                    static_cast<u32>(sizeof(RescalingLayout) + sizeof(RenderAreaLayout))};
    return Vk().device->GetLogical().CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = GplOn() ? VkPipelineLayoutCreateFlags{VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT} : 0u,
        .setLayoutCount = static_cast<u32>(sets.size()),
        .pSetLayouts = sets.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
}

struct StageCache {
    std::mutex mu;
    std::unordered_map<std::string, std::vector<std::shared_ptr<StageBuild>>> stages;
};
StageCache& Stages() {
    static StageCache c;
    return c;
}

bool StageHolds(const StageBuild& b, const RenderState& rs) {
    for (const auto& r : b.cbuf_reads) {
        if (CbufWord(rs, r.nvn_stage, r.bank, r.offset) != r.value) return false;
    }
    for (const auto& r : b.type_reads) {
        if (!TypeHolds(r, rs)) return false;
    }
    return true;
}

template <typename T>
void PutBytes(std::string& key, const T& v) {
    key.append(reinterpret_cast<const char*>(&v), sizeof v);
}

/// The runtime state a stage translated alone is keyed by (StageFor's key; the shader store's assumptions part).
std::string StageAssumptions(u32 s, const Shader::RuntimeInfo& rt, const std::string& extra) {
    std::string key;
    if (s == kVertex) PutBytes(key, rt.generic_input_types);
    if (s == kFragment) PutBytes(key, rt.color_output_types);
    PutBytes(key, rt.convert_depth_mode);
    PutBytes(key, rt.input_topology);
    return key + extra;
}

/// Whether a stored stage's checks hold in `rs` (as a variant's: StillHolds).
bool StoredHolds(const StoredStage& st, const RenderState& rs) {
    for (const auto& r : st.cbuf_reads) {
        if (CbufWord(rs, r.nvn_stage, r.bank, r.offset) != r.value) return false;
    }
    for (const auto& r : st.type_reads) {
        if (!TypeHolds(r, rs)) return false;
    }
    return true;
}

/// RT_SHADER_STORE_CHECK: a stored stage against a fresh translation of it (the reference).
void CheckStored(const char* what, u64 hash, const StoredStage& st, const std::vector<u32>& spirv, const Shader::Info& info) {
    static std::atomic<u64> compared{0}, differ{0};
    const bool same_spirv = st.spirv == spirv, same_info = InfoBytes(st.info) == InfoBytes(info);
    if ((!same_spirv || !same_info) && differ.fetch_add(1) < 40) {
        Log("nvn: SHADER STORE CHECK: the stored %s stage %016llx differs from a fresh translation: %s%s (stored %zu words, "
            "fresh %zu)%s%s", what, (unsigned long long)hash, same_spirv ? "" : "SPIR-V ", same_info ? "" : "Info",
            st.spirv.size(), spirv.size(), same_info ? "" : "; Info: ", same_info ? "" : InfoDifference(st.info, info).c_str());
    }
    const u64 n = compared.fetch_add(1) + 1;
    if (n == 1 || n == 10 || n % 100 == 0) {
        Log("nvn: shader store check: %llu stored stages compared with a fresh translation, %llu differ",
            (unsigned long long)n, (unsigned long long)differ.load());
    }
}

/// drippu's translator runs on one thread at a time (it is not known to be re-entrant): StageFor, the seed.
std::mutex& TranslatorLock() {
    static std::mutex m;
    return m;
}

void FinishStage(StageBuild& b, u32 s, const std::vector<u32>& spirv);

/// Translates stage `s` (code memory `code`) alone with runtime state `rt` and `hooks` into `b`: its module, info,
/// descriptor set and templates. `building_cfg` is cleared once the control flow is built (ReadsThatPin).
/// `spirv_out`: also gets the SPIR-V as bound (the shader store, SHADER_PLAN Phase 4).
void TranslateStageInto(StageBuild& b, const std::vector<u8>& code, const ProgramStage& program, u32 s,
                        const Shader::RuntimeInfo& rt, ShaderHooks hooks, const std::shared_ptr<bool>& building_cfg,
                        std::vector<u32>* spirv_out = nullptr) {
    std::lock_guard translating(TranslatorLock());
    NvnEnvironment env(code, program, std::move(hooks));
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
    Shader::ObjectPool<Shader::IR::Inst> insts;
    Shader::ObjectPool<Shader::IR::Block> blocks;
    Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, env.StartAddress() + static_cast<u32>(sizeof(Shader::ProgramHeader)), false);
    *building_cfg = false;
    Shader::IR::Program ir = Shader::Maxwell::TranslateProgram(insts, blocks, env, cfg, ShaderHostInfo());
    RescalingPass(ir);
    Shader::RuntimeInfo copy = rt;
    Shader::Maxwell::ConvertLegacyToGeneric(ir, copy);
    DumpStage(program, ir);
    Shader::Backend::Bindings bindings;
    SetRescaleBase(bindings, ir.stage);
    std::vector<u32> spirv = Shader::Backend::SPIRV::EmitSPIRV(ShaderProfile(), copy, ir, bindings);
    if (SetOf(s)) MoveDescriptorSet(spirv, SetOf(s));
    if (s != kFragment) MakePositionInvariant(spirv);
    if (const char* dir = std::getenv("RT_NVN_DUMP_SPIRV")) { // <stage hash>_<stage>.spv, for comparing (Phase 5)
        char name[48];
        std::snprintf(name, sizeof name, "%016llx_%u.spv", (unsigned long long)CodeHash(code, program), s);
        std::filesystem::create_directories(dir);
        std::ofstream(std::filesystem::path(dir) / name, std::ios::binary)
            .write(reinterpret_cast<const char*>(spirv.data()), static_cast<std::streamsize>(spirv.size() * 4));
    }
    b.info = ir.info;
    FinishStage(b, s, spirv);
    if (spirv_out) *spirv_out = std::move(spirv);
}

/// A translated stage's Vulkan objects from its SPIR-V and `b.info`: module, set layout, templates (a fresh
/// translation's or a stored one's).
void FinishStage(StageBuild& b, u32 s, const std::vector<u32>& spirv) {
    const auto& device = *Vk().device;
    b.module = device.GetLogical().CreateShaderModule({
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = spirv.size() * 4,
        .pCode = spirv.data(),
    });
    Vulkan::DescriptorLayoutBuilder builder{device};
    builder.Add(b.info, VkStage(s));
    b.push = s == kFragment && UsePush(builder);
    b.set_layout = builder.CreateDescriptorSetLayout(b.push);
    if (b.push) {
        // the template is made for a layout of set 0 alone, compatible with every pipeline's (independent sets
        // as theirs when linked from libraries)
        const VkDescriptorSetLayout set = *b.set_layout;
        b.push_layout = PipelineLayout({&set, 1});
        b.push_update = builder.CreateTemplate(*b.set_layout, *b.push_layout, true);
    } else {
        b.set_update = builder.CreateTemplate(*b.set_layout, VK_NULL_HANDLE, false);
    }
}

/// Stage `s` of `program` translated alone with runtime state `rt`, from the cache or translated now; `extra`
/// is a key part beyond `rt`'s assumptions (a fragment stage built for its previous stage).
std::shared_ptr<const StageBuild> StageFor(const ProgramStage& program, u32 s, const RenderState& rs,
                                           const Shader::RuntimeInfo& rt, const std::string& extra) {
    const std::vector<u8> code = ReadCode(program);
    std::string key(reinterpret_cast<const char*>(code.data()), code.size());
    PutBytes(key, s);
    PutBytes(key, std::array<u32, 4>{program.program_offset, program.program_size, program.constants_offset,
                                     program.constants_size});
    const std::string assumptions = StageAssumptions(s, rt, extra);
    key += assumptions;

    StageCache& c = Stages();
    std::lock_guard lock(c.mu);
    auto& variants = c.stages[key];
    for (const auto& b : variants) {
        if (StageHolds(*b, rs)) {
            TheTotals().stage_reuses.fetch_add(1, std::memory_order_relaxed);
            return b;
        }
    }
    // translated before play (the seed's, SHADER_PLAN Phase 4) and its checks hold: no translation
    std::optional<StoredStage> stored;
    if (ShaderStoreOn()) {
        stored = StoreGet(StoreKey(code, program, s, assumptions));
        if (stored && !StoredHolds(*stored, rs)) stored.reset();
        if (stored && !ShaderStoreCheck()) {
            auto b = std::make_shared<StageBuild>();
            b->nvn_stage = s;
            b->info = stored->info;
            b->cbuf_reads = stored->cbuf_reads;
            b->type_reads = stored->type_reads;
            FinishStage(*b, s, stored->spirv);
            TheTotals().stages_stored.fetch_add(1, std::memory_order_relaxed);
            variants.push_back(b);
            return b;
        }
    }
    auto b = std::make_shared<StageBuild>();
    b->nvn_stage = s;
    auto building_cfg = std::make_shared<bool>(true);
    auto recorder = std::make_shared<TypeRecorder>();
    ShaderHooks hooks{
        .cbuf = [&b, &rs, s, building_cfg, recorder](u32 bank, u32 offset) {
            const u32 v = CbufWord(rs, s, bank, offset);
            if (*building_cfg) b->cbuf_reads.push_back({s, bank, offset, v}); // see ReadsThatPin
            else recorder->Word(bank, offset, v);
            return v;
        },
        .texture_type = [&b, &rs, s, recorder](u32 handle) {
            b->type_reads.push_back(recorder->Type(rs, s, handle));
            return b->type_reads.back().type;
        },
    };
    std::vector<u32> spirv;
    TranslateStageInto(*b, code, program, s, rt, std::move(hooks), building_cfg, stored ? &spirv : nullptr);
    if (stored) CheckStored("graphics", CodeHash(code, program), *stored, spirv, b->info);
    TheTotals().stages.fetch_add(1, std::memory_order_relaxed);
    variants.push_back(b);
    return b;
}

/// Whether fragment stage `fs` (translated alone) reads only what `previous` writes: generic inputs it stores,
/// and no legacy input (those are placed where the previous stage put them).
bool InputsWritten(const Shader::Info& fs, const Shader::Info& previous) {
    if (fs.loads.Legacy()) return false;
    for (size_t i = 0; i < Shader::IR::NUM_GENERICS; ++i) {
        if (fs.loads.Generic(i) && !previous.stores.Generic(i)) return false;
    }
    return true;
}

/// A set layout without bindings, for a pipeline layout's unused set numbers.
VkDescriptorSetLayout EmptySetLayout() {
    static const Vulkan::vk::DescriptorSetLayout empty = Vk().device->GetLogical().CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = 0,
        .pBindings = nullptr,
    });
    return *empty;
}

/// Build's shaders with RT_NVN_STAGE_ALONE: fills `p`'s stages, layout and reads, and `stage_infos`; returns
/// whether there is a fragment stage.
bool BuildStagesAlone(GraphicsPipeline& p, const RenderState& rs, u32 mode, const TargetSet& targets, bool gl_ndc,
                      std::vector<VkPipelineShaderStageCreateInfo>& stage_infos) {
    if (!rs.stages[kVertex]) Unsupported("a draw without a vertex program");
    const bool has_geometry = rs.stages[kGeometry].has_value();
    std::vector<VkDescriptorSetLayout> sets(has_geometry ? 3 : 2, EmptySetLayout());
    std::shared_ptr<const StageBuild> previous;
    bool has_fragment = false;
    for (u32 s : kPipelineOrder) {
        if (!rs.stages[s]) continue;
        if (s == kTessControl || s == kTessEvaluation) Unsupported("tessellation");
        const ProgramStage& program = *rs.stages[s];
        Shader::RuntimeInfo rt;
        rt.previous_stage_stores.mask.set();
        switch (s) {
        case kVertex:
            if (!has_geometry) rt.convert_depth_mode = gl_ndc;
            for (u32 i = 0; i < rs.vertex_attrib_count; ++i) {
                const u32 f = rs.vertex_attribs[i] & 0xFF;
                if (f) rt.generic_input_types[i] = FormatType(VertexFormat(f));
            }
            break;
        case kGeometry:
            rt.convert_depth_mode = gl_ndc;
            rt.input_topology = GeometryInput(mode);
            break;
        case kFragment: {
            const ColorState cs = std::bit_cast<ColorState>(rs.color);
            if (cs.alpha_test != 8) Unsupported("alpha test %u", cs.alpha_test); // needs the reference value
            for (u32 i = 0; i < targets.color_count; ++i) {
                if (targets.colors[i] != VK_FORMAT_UNDEFINED) rt.color_output_types[i] = FormatType(targets.colors[i]);
            }
            break;
        }
        }
        std::shared_ptr<const StageBuild> b = StageFor(program, s, rs, rt, {});
        Shader::RuntimeInfo actual = rt; // what the stage gets from its previous one (RT_NVN_STAGE_DEPS)
        if (previous) {
            actual.previous_stage_stores = previous->info.stores;
            actual.previous_stage_legacy_stores_mapping = previous->info.legacy_stores_mapping;
        }
        if (s == kFragment && previous && !InputsWritten(b->info, previous->info)) {
            std::string extra(reinterpret_cast<const char*>(&previous->info.stores.mask), sizeof previous->info.stores.mask);
            for (const auto& [from, to] : previous->info.legacy_stores_mapping) PutBytes(extra, std::pair{from, to});
            static std::atomic<u32> told{0};
            if (told.fetch_add(1) < 10) {
                std::string missing;
                for (size_t i = 0; i < Shader::IR::NUM_GENERICS; ++i) {
                    if (b->info.loads.Generic(i) && !previous->info.stores.Generic(i)) missing += " " + std::to_string(i);
                }
                Log("nvn: fragment stage %016llx reads %s%s%s that its previous stage does not write: translated for it",
                    (unsigned long long)StageHash(program), missing.empty() ? "" : "generic input(s)", missing.c_str(),
                    b->info.loads.Legacy() ? (missing.empty() ? "legacy inputs" : " and legacy inputs") : "");
            }
            b = StageFor(program, s, rs, actual, extra);
            TheTotals().stages_for_previous.fetch_add(1, std::memory_order_relaxed);
        }
        p.cbuf_reads.insert(p.cbuf_reads.end(), b->cbuf_reads.begin(), b->cbuf_reads.end());
        p.type_reads.insert(p.type_reads.end(), b->type_reads.begin(), b->type_reads.end());
        if (StageDepsOn()) LogStageDeps(p, program, s, rs, actual);
        stage_infos.push_back({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VkStage(s),
            .module = *b->module,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        });
        p.stages.push_back({s, program, b->info, SetOf(s), b});
        if (b->set_layout) sets[SetOf(s)] = *b->set_layout;
        p.uses_render_area |= b->info.uses_render_area;
        if (s == kFragment) {
            has_fragment = true;
            p.push = b->push;
        } else {
            previous = b;
        }
    }
    p.layout = PipelineLayout(sets);
    return has_fragment;
}

} // namespace

/// One part of a linked graphics pipeline (RT_NVN_GPL, pipeline.h): its own layout (shader libraries: their
/// stages' sets only) and the library.
struct PipelineLibrary {
    Vulkan::vk::PipelineLayout layout;
    Vulkan::vk::Pipeline pipeline;
};

namespace {

// Each library holds only the dynamic state of its part (the rest is set at each draw as well, draws.cpp).
constexpr VkDynamicState kPreRasterDynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE_EXT, VK_DYNAMIC_STATE_FRONT_FACE_EXT,
    VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT, VK_DYNAMIC_STATE_DEPTH_BIAS,
};
constexpr VkDynamicState kFragmentDynamicStates[] = {
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    VK_DYNAMIC_STATE_DEPTH_BOUNDS, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
    VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
    VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT, VK_DYNAMIC_STATE_STENCIL_OP_EXT,
};
constexpr VkDynamicState kOutputDynamicStates[] = {VK_DYNAMIC_STATE_BLEND_CONSTANTS};
constexpr VkPipelineDynamicStateCreateInfo kPreRasterDynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                                             std::size(kPreRasterDynamicStates), kPreRasterDynamicStates};
constexpr VkPipelineDynamicStateCreateInfo kFragmentDynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                                            std::size(kFragmentDynamicStates), kFragmentDynamicStates};
constexpr VkPipelineDynamicStateCreateInfo kOutputDynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                                          std::size(kOutputDynamicStates), kOutputDynamicStates};

struct Libraries {
    struct Slot { // one library, made by the first builder that needs it while others wait on `mu`
        std::mutex mu;
        std::shared_ptr<const PipelineLibrary> lib;
    };
    std::mutex mu;
    std::map<std::string, std::shared_ptr<Slot>> parts;
    std::map<const StageBuild*, std::set<VkRenderPass>> passes; // the render passes a stage build's libraries are for
    std::atomic<u64> made{0}, reused{0}, links{0};
};
Libraries& TheLibraries() {
    static Libraries l;
    return l;
}

/// The library for `part` under `key`, made at first use (`fill` adds the part's state; `sets`: its layout's sets,
/// empty for the two parts without shaders); `hit` is cleared when the driver's cache did not have a new one.
template <typename F>
std::shared_ptr<const PipelineLibrary> LibraryFor(VkGraphicsPipelineLibraryFlagsEXT part, const std::string& key,
                                                  VkRenderPass pass, std::span<const VkDescriptorSetLayout> sets,
                                                  const VkPipelineDynamicStateCreateInfo& dynamic, bool& hit, F&& fill) {
    Libraries& l = TheLibraries();
    std::shared_ptr<Libraries::Slot> slot;
    {
        std::lock_guard lock(l.mu);
        auto& s = l.parts[key];
        if (!s) s = std::make_shared<Libraries::Slot>();
        slot = s;
    }
    std::lock_guard made(slot->mu); // builders of different libraries compile at the same time (Phase 3)
    if (slot->lib) {
        l.reused.fetch_add(1, std::memory_order_relaxed);
        return slot->lib;
    }
    auto lib = std::make_shared<PipelineLibrary>();
    if (!sets.empty()) lib->layout = PipelineLayout(sets);
    VkPipelineCreationFeedback feedback{};
    const VkPipelineCreationFeedbackCreateInfo feedback_info{VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
                                                             nullptr, &feedback, 0, nullptr};
    const VkGraphicsPipelineLibraryCreateInfoEXT info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
                                                      FeedbackOn() ? &feedback_info : nullptr, part};
    VkGraphicsPipelineCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &info,
        .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
        .pDynamicState = dynamic.dynamicStateCount ? &dynamic : nullptr,
        .layout = lib->layout ? *lib->layout : VK_NULL_HANDLE,
        .renderPass = part == VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT ? VK_NULL_HANDLE : pass,
        .subpass = 0,
    };
    fill(ci);
    lib->pipeline = Vk().device->GetLogical().CreateGraphicsPipeline(ci, DiskCache());
    if ((feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) &&
        !(feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT)) {
        hit = false;
    }
    l.made.fetch_add(1, std::memory_order_relaxed);
    slot->lib = lib;
    return lib;
}

void NoteLibraryPass(const StageBuild* b, VkRenderPass pass) {
    if (!b) return;
    Libraries& l = TheLibraries();
    std::lock_guard lock(l.mu);
    l.passes[b].insert(pass);
}

/// `p`'s pipeline linked from its libraries: fast, or optimised (link-time optimisation, the worker's).
Vulkan::vk::Pipeline Link(const GraphicsPipeline& p, VkRenderPass pass, bool optimised) {
    std::vector<VkPipeline> parts;
    for (const auto& l : p.libraries) parts.push_back(*l->pipeline);
    const VkPipelineLibraryCreateInfoKHR libraries{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, nullptr,
                                                   static_cast<u32>(parts.size()), parts.data()};
    TheLibraries().links.fetch_add(1, std::memory_order_relaxed);
    return Vk().device->GetLogical().CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &libraries,
        .flags = optimised ? VkPipelineCreateFlags{VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT} : 0u,
        .layout = *p.layout,
        .renderPass = pass,
        .subpass = 0,
    }, DiskCache());
}

// RT_NVN_GPL_OPTIMISE=1 (default off): every linked pipeline is linked again with link-time optimisation on a worker
// thread and drawn with from then on (GraphicsPipeline::optimised); the fast link stays alive (command buffers in
// flight may use it). Off since 2026-10-23: making those links while the game loaded the plaza held its loading
// thread ~1.1 s longer in 5 of 7 runs (0 of 2 without; also with the re-links built but never drawn), and the
// fast links draw as fast at 720p (the GPU is not the limit). To measure again at 1440p (PORT_PLAN B4), e.g. with
// the re-links deferred to idle time.
bool OptimiseOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_NVN_GPL_OPTIMISE");
        const bool optimise = e && std::strcmp(e, "0") != 0;
        if (optimise) Log("nvn: RT_NVN_GPL_OPTIMISE: linked pipelines re-linked with link-time optimisation");
        return optimise;
    }();
    return on;
}

class Optimiser {
public:
    void Add(GraphicsPipeline& p, VkRenderPass pass) {
        std::lock_guard lock(mu_);
        if (!started_) {
            std::thread([this] { Run(); }).detach(); // runs until the process ends
            started_ = true;
        }
        jobs_.push_back({&p, pass});
        cv_.notify_one();
    }
    std::atomic<u64> done{0}, ns{0}; // optimised links made, and their time

private:
    struct Job {
        GraphicsPipeline* p; // pipelines live until the process ends (Pipelines())
        VkRenderPass pass;
    };
    void Run() {
        for (;;) {
            Job job;
            {
                std::unique_lock lock(mu_);
                cv_.wait(lock, [&] { return !jobs_.empty(); });
                job = jobs_.front();
                jobs_.pop_front();
            }
            const auto t0 = std::chrono::steady_clock::now();
            try {
                job.p->optimised_owner = Link(*job.p, job.pass, true);
                job.p->optimised.store(*job.p->optimised_owner, std::memory_order_release);
            } catch (const std::exception& e) {
                Log("nvn: optimised link failed: %s (the fast link stays)", e.what());
            }
            ns.fetch_add(static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
            done.fetch_add(1);
        }
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> jobs_;
    bool started_ = false;
};
Optimiser& TheOptimiser() {
    static Optimiser o;
    return o;
}

/// "; L libraries (R reused), K links; stage builds with libraries for several render passes: S of T; O optimised
/// links (X ms)" when linking from libraries, else "".
std::string LibraryTotals() {
    if (!GplOn()) return "";
    Libraries& l = TheLibraries();
    size_t several = 0, stages = 0;
    {
        std::lock_guard lock(l.mu);
        stages = l.passes.size();
        for (const auto& [b, passes] : l.passes) several += passes.size() > 1;
    }
    const Optimiser& o = TheOptimiser();
    char b[224];
    std::snprintf(b, sizeof b,
                  "; %llu libraries (%llu reused), %llu links; stage builds with libraries for several render passes: "
                  "%zu of %zu; %llu optimised links (%.0f ms)",
                  (unsigned long long)l.made.load(), (unsigned long long)l.reused.load(), (unsigned long long)l.links.load(),
                  several, stages, (unsigned long long)o.done.load(), o.ns.load() / 1e6);
    return b;
}

/// Translates compute stage `p.program` (its code memory `code`; `hooks` answer its questions, `building_cfg` true
/// until its control-flow graph is built) and creates its pipeline through `cache`: play's BuildCompute and the
/// seed's SeedStage, so both make the same SPIR-V, layout and pipeline.
void FinishCompute(ComputePipeline& p, const std::vector<u32>& spirv, VkPipelineCache cache);

/// `spirv_out`: also gets the SPIR-V (the shader store, SHADER_PLAN Phase 4).
void MakeCompute(ComputePipeline& p, std::vector<u8> code, ShaderHooks hooks, bool& building_cfg, VkPipelineCache cache,
                 std::vector<u32>* spirv_out = nullptr) {
    const u64 hash = CodeHash(code, p.program);
    NvnEnvironment env(std::move(code), p.program, std::move(hooks));
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
    Shader::ObjectPool<Shader::IR::Inst> insts;
    Shader::ObjectPool<Shader::IR::Block> blocks;
    Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, env.StartAddress(), false);
    building_cfg = false;
    Shader::IR::Program ir = TranslateComputeProgram(insts, blocks, env, cfg, ShaderHostInfo());
    RescalingPass(ir);
    DumpStage(p.program, ir);
    std::vector<u32> spirv = Shader::Backend::SPIRV::EmitSPIRV(ShaderProfile(), ir);
    if (const char* dir = std::getenv("RT_NVN_DUMP_SPIRV")) { // as TranslateStageInto's: <stage hash>_5.spv
        char name[48];
        std::snprintf(name, sizeof name, "%016llx_5.spv", (unsigned long long)hash);
        std::filesystem::create_directories(dir);
        std::ofstream(std::filesystem::path(dir) / name, std::ios::binary)
            .write(reinterpret_cast<const char*>(spirv.data()), static_cast<std::streamsize>(spirv.size() * 4));
    }
    p.info = ir.info;
    FinishCompute(p, spirv, cache);
    if (spirv_out) *spirv_out = std::move(spirv);
}

/// A compute pipeline's Vulkan objects from its SPIR-V and `p.info` (a fresh translation's or a stored one's).
void FinishCompute(ComputePipeline& p, const std::vector<u32>& spirv, VkPipelineCache cache) {
    const auto& device = *Vk().device;
    const auto& logical = device.GetLogical();
    p.module = logical.CreateShaderModule({
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = spirv.size() * 4,
        .pCode = spirv.data(),
    });
    Vulkan::DescriptorLayoutBuilder builder{device};
    builder.Add(p.info, VK_SHADER_STAGE_COMPUTE_BIT);
    p.push = UsePush(builder);
    p.set_layout = builder.CreateDescriptorSetLayout(p.push);
    p.layout = builder.CreatePipelineLayout(*p.set_layout);
    p.update = builder.CreateTemplate(*p.set_layout, *p.layout, p.push);
    PipelineTimer timer;
    p.pipeline = logical.CreateComputePipeline({
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = timer.Next(),
        .flags = 0,
        .stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *p.module,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        .layout = *p.layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    }, cache);
}

// RT_NVN_SEED=<dir> (SHADER_PLAN Phase 5): the game's shader stages, extracted from the user's own game files
// (<hash>.ctrl / .code / .asm: tools/shaderseed), are each built once into a stage library on worker threads -
// translated as StageFor translates them, so the SPIR-V is what play makes, then compiled by the driver into our
// pipeline cache file (DiskCache; NVIDIA's own disk cache until 2026-10-24, which holds ~1 GB in all and dropped
// part of the ~1.2 GB seed) - and the library dropped. A draw in play then finds its stages' compiles in that cache
// (on NVIDIA a stage library's cache entry does not depend on the render pass: measured 2026-10-23, 0.2 ms for
// another pass in a later process). The marker is written with the cache file that holds the seed (WriteCache). What StageFor takes from the bound state comes from the stage's assembly
// text: integer vertex attributes (read by .S/.U or I2F instructions; unsigned, as the game's formats are),
// texture types (buffer when the text has BUFFER targets, else 2D: no stage of 45,541 mixes them), float colour
// outputs, the device's depth mode. Compute stages are built as whole pipelines (MakeCompute, as play's
// BuildCompute; since 2026-10-24: the new-game run's only build hitches were cold compute builds, 30-56 ms). Stages
// whose control flow reads constant words (unknown values) and geometry stages are left to play. Started at the game's first graphics pipeline, in the background; a marker
// next to the pipeline cache (<program id>.nvn.seed: driver, device, stage count) skips it once done (and only
// while that cache file exists).
// RT_NVN_SEED_ONLY=1: the seed alone at the first NVN call, waiting for it, then the run ends (depth mode
// RT_NVN_SEED_DEPTH_MODE, default 1: Brothership's).

/// The integer generic attributes a vertex stage's assembly text reads (as integers: .S/.U types, I2F).
std::bitset<32> IntegerAttributes(const std::string& text) {
    std::bitset<32> ints;
    size_t line_start = 0;
    while (line_start < text.size()) {
        const size_t end = std::min(text.find('\n', line_start), text.size());
        const std::string_view line(text.data() + line_start, end - line_start);
        line_start = end + 1;
        const size_t op_start = line.find_first_not_of(" \t");
        if (op_start == std::string_view::npos) continue;
        const std::string_view op = line.substr(op_start, line.find_first_of(" \t", op_start) - op_start);
        const bool integer = op.starts_with("I2F") || op.find(".S") != std::string_view::npos || op.find(".U") != std::string_view::npos;
        if (!integer) continue;
        for (size_t at = line.find("vertex.attrib["); at != std::string_view::npos; at = line.find("vertex.attrib[", at + 1)) {
            const u32 index = static_cast<u32>(std::strtoul(line.data() + at + 14, nullptr, 10));
            if (index < 32) ints.set(index);
        }
    }
    return ints;
}

/// The texture type of each bindless handle a stage's assembly text uses, keyed by its words' c2 byte offsets OR-ed
/// (the seed's own handle values are decoded to that key: SeedTextKey): the text loads handle words with
/// `LDC.U64 Dn.x, buf14[N]` (c2 offset N + 32: checked against drippu's tracking, 2026-10-24), combines texture and
/// sampler words with `OR.U64`, and names each texture instruction's target last (`..., handle(D0.y), 3D;`).
/// A value used with two targets is left out (answered by the old rule). Brothership's stages mix targets often
/// (2D with 3D in 14,743 of 45,792), so one answer per stage was wrong for some (a 3D compute stage seeded as 2D).
std::map<u32, Shader::TextureType> TextureTargets(const std::string& text) {
    static const std::map<std::string_view, Shader::TextureType> kTargets{
        {"1D", Shader::TextureType::Color1D},         {"ARRAY1D", Shader::TextureType::ColorArray1D},
        {"2D", Shader::TextureType::Color2D},         {"ARRAY2D", Shader::TextureType::ColorArray2D},
        {"3D", Shader::TextureType::Color3D},         {"CUBE", Shader::TextureType::ColorCube},
        {"ARRAYCUBE", Shader::TextureType::ColorArrayCube}, {"BUFFER", Shader::TextureType::Buffer},
        {"RECT", Shader::TextureType::Color2DRect},   {"2DMS", Shader::TextureType::Color2D},
        {"ARRAY2DMS", Shader::TextureType::ColorArray2D},
    };
    const auto reg = [](std::string_view r) { // "D0" -> "D0.x"; anything else as it is
        while (!r.empty() && (r.front() == '-' || r.front() == '|')) r.remove_prefix(1);
        std::string s(r);
        if (s.size() >= 2 && s[0] == 'D' && s.find('.') == std::string::npos) s += ".x";
        return s;
    };
    std::map<std::string, u32> value; // register -> handle value
    std::map<u32, Shader::TextureType> types;
    std::set<u32> mixed;
    size_t line_start = 0;
    while (line_start < text.size()) {
        const size_t end = std::min(text.find('\n', line_start), text.size());
        std::string_view line(text.data() + line_start, end - line_start);
        line_start = end + 1;
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.remove_prefix(1);
        if (const size_t semi = line.rfind(';'); semi != std::string_view::npos) line = line.substr(0, semi);
        else continue;
        const size_t sp = line.find(' ');
        if (sp == std::string_view::npos) continue;
        const std::string_view op = line.substr(0, sp);
        std::vector<std::string_view> args;
        for (std::string_view rest = line.substr(sp + 1); !rest.empty();) {
            const size_t comma = rest.find(", ");
            args.push_back(rest.substr(0, comma));
            rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 2);
        }
        if (const size_t h = line.find("handle("); h != std::string_view::npos) { // a texture or image instruction
            const size_t close = line.find(')', h);
            const auto it = value.find(reg(line.substr(h + 7, close - h - 7)));
            const auto target = kTargets.find(args.back());
            if (it != value.end() && target != kTargets.end() && !mixed.count(it->second)) {
                const auto [at, fresh] = types.emplace(it->second, target->second);
                if (!fresh && at->second != target->second) types.erase(at), mixed.insert(it->second);
            }
            continue;
        }
        if (args.empty() || args[0].empty() || args[0][0] != 'D') continue;
        const std::string dst = reg(args[0]);
        value.erase(dst); // written: known again only as below
        if (op == "LDC.U64" && args.size() == 2 && args[1].starts_with("buf14[")) {
            value[dst] = static_cast<u32>(std::strtoul(args[1].data() + 6, nullptr, 10)) + 32;
        } else if (op == "OR.U64" && args.size() == 3) {
            const auto a = value.find(reg(args[1])), b = value.find(reg(args[2]));
            if (a != value.end() && b != value.end()) value[dst] = a->second | b->second;
        } else if (op == "MOV.U64" && args.size() == 2) {
            if (const auto a = value.find(reg(args[1])); a != value.end()) value[dst] = a->second;
        }
    }
    return types;
}

/// The seed's answer to a constant-buffer read: each distinct c2 word of the stage a bit of its own (1, 2, 4, ... in
/// the order read), others 0. So a handle - one word, or two OR-ed - tells which words it came from (the shader
/// store's checks, SeedStore); byte offsets, as before 2026-10-28, OR-ed look like one another (0x20 | 0x24 is
/// 0x24). The SPIR-V does not depend on the values (only the texture types answered do: SeedTextKey). Per stage.
struct SeedWords {
    std::vector<u32> offsets; // bit i: c2 offset offsets[i]
    bool overflow = false;    // more than 32 distinct words: answered 0 past them (not stored)
};
thread_local SeedWords t_seed_words;

u32 SeedCbuf(u32 bank, u32 offset) {
    if (bank != 2) return 0;
    SeedWords& w = t_seed_words;
    for (size_t i = 0; i < w.offsets.size(); ++i) {
        if (w.offsets[i] == offset) return 1u << i;
    }
    if (w.offsets.size() == 32) {
        w.overflow = true;
        return 0;
    }
    w.offsets.push_back(offset);
    return 1u << (w.offsets.size() - 1);
}

/// A seed handle's key in TextureTargets' map: the byte offsets of its words, OR-ed as the text's values are.
u32 SeedTextKey(u32 handle) {
    u32 key = 0;
    const SeedWords& w = t_seed_words;
    for (size_t i = 0; i < w.offsets.size(); ++i) {
        if (handle >> i & 1) key |= w.offsets[i];
    }
    return key;
}

std::vector<u8> ReadAll(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

struct Seed {
    std::filesystem::path dir;
    std::vector<std::string> names; // the stages' <hash>
    bool gl_ndc = false;
    std::atomic<size_t> next{0}, built{0}, left{0}, failed{0}, finished{0};
    std::atomic<size_t> types_from_text{0}, types_guessed{0}; // texture types asked: found by TextureTargets, or not
    std::atomic<size_t> cache_hits{0}; // pipelines our pipeline cache already had (creation feedback)
    std::atomic<size_t> stored{0}, unstorable{0}; // into the shader store; not (a texture handle of unknown source)
    std::atomic<u64> driver_ns{0}, create_ns{0}; // all the work; the driver's library creations alone
    std::filesystem::path marker;
    std::string tag;
    std::chrono::steady_clock::time_point start;
    u32 threads = 0;
};

/// A seed stage's translation into the shader store (SHADER_PLAN Phase 4), each texture type it was answered with
/// stored with the constant words its handle came from (decoded from SeedCbuf's bits), so play checks it as a
/// variant's (TypeHolds: the words OR-ed). Not stored: more than 32 handle words, a handle of more than two words,
/// a descriptor with shifted words or an array (TypeHolds could not rebuild the handle).
void SeedStore(Seed& seed, const std::string& key, std::vector<u32> spirv, const Shader::Info& info,
               std::vector<GraphicsPipeline::TypeRead> types) {
    if (!ShaderStoreOn()) return;
    const SeedWords& w = t_seed_words;
    bool ok = !w.overflow;
    for (const auto& d : info.texture_descriptors) ok &= d.shift_left == 0 && d.secondary_shift_left == 0 && d.count == 1;
    for (const auto& d : info.texture_buffer_descriptors) ok &= d.shift_left == 0 && d.secondary_shift_left == 0 && d.count == 1;
    for (const auto& d : info.image_descriptors) ok &= d.count == 1;
    for (const auto& d : info.image_buffer_descriptors) ok &= d.count == 1;
    for (auto& t : types) {
        if (!ok) break;
        std::vector<u32> words;
        for (u32 i = 0; i < 32; ++i) {
            if (!(t.handle >> i & 1)) continue;
            if (i >= w.offsets.size()) ok = false;
            else words.push_back(w.offsets[i]);
        }
        if (words.empty() || words.size() > 2) ok = false;
        if (!ok) break;
        t.words = static_cast<u32>(words.size()), t.bank = 2, t.offset = words[0];
        if (words.size() == 2) t.bank2 = 2, t.offset2 = words[1];
    }
    if (!ok) {
        seed.unstorable.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    StorePut(key, StoredStage{std::move(spirv), info, {}, std::move(types)});
    seed.stored.fetch_add(1, std::memory_order_relaxed);
}

/// One seed stage: translated and built into a library, then dropped. False: left to play.
bool SeedStage(Seed& seed, const std::string& name, VkRenderPass pass) {
    const std::vector<u8> ctrl = ReadAll(seed.dir / (name + ".ctrl"));
    std::vector<u8> code = ReadAll(seed.dir / (name + ".code"));
    const std::vector<u8> asm_bytes = ReadAll(seed.dir / (name + ".asm"));
    if (ctrl.size() < kControlSize) return false;
    std::string why;
    const auto program = ParseControlBytes(ctrl.data(), 0, why);
    if (!program) return false;
    const u32 s = static_cast<u32>(program->stage);
    t_seed_words = {}; // this stage's handle words (SeedCbuf)
    const std::string text(asm_bytes.begin(), asm_bytes.end());
    const Shader::TextureType texture = text.find(", BUFFER;") != std::string::npos ? Shader::TextureType::Buffer
                                                                                   : Shader::TextureType::Color2D;
    const auto targets = std::make_shared<std::map<u32, Shader::TextureType>>(TextureTargets(text));
    const auto type_of = [targets, texture, &seed](u32 handle) { // by handle; else the stage's old rule
        if (const auto it = targets->find(SeedTextKey(handle)); it != targets->end()) {
            seed.types_from_text.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
        seed.types_guessed.fetch_add(1, std::memory_order_relaxed);
        return texture;
    };
    if (program->stage == Stage::kCompute) { // a whole compute pipeline, as play's BuildCompute makes it
        ComputePipeline p;
        p.program = *program;
        bool pinned = false; // the control-flow graph alone first: a stage whose control flow reads constant words
        {                    // (unknown here) is left to play, before any compile
            NvnEnvironment env(std::vector<u8>(code), p.program, ShaderHooks{
                .cbuf = [&pinned](u32, u32) { pinned = true; return 0u; },
                .texture_type = type_of,
            });
            Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
            Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, env.StartAddress(), false);
        }
        if (pinned) return false;
        bool building_cfg = true;
        std::vector<GraphicsPipeline::TypeRead> types; // what each texture type was answered with (SeedStore)
        ShaderHooks hooks{
            .cbuf = SeedCbuf, // handle words for the texture pass
            .texture_type = [&types, &type_of](u32 handle) {
                types.push_back({5, handle, type_of(handle)});
                return types.back().type;
            },
        };
        const std::vector<u8> code_copy = code;
        std::vector<u32> spirv;
        const auto t0 = std::chrono::steady_clock::now();
        MakeCompute(p, std::move(code), std::move(hooks), building_cfg, DiskCache(), &spirv); // into our pipeline cache
        if (t_cache_hit == 1) seed.cache_hits.fetch_add(1, std::memory_order_relaxed);
        SeedStore(seed, StoreKey(code_copy, p.program, 5, std::string()), std::move(spirv), p.info, std::move(types));
        seed.create_ns.fetch_add(static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
        return true;
    }
    if (s != kVertex && s != kFragment) return false; // geometry needs the draw's topology
    Shader::RuntimeInfo rt;
    rt.previous_stage_stores.mask.set();
    if (s == kVertex) {
        rt.convert_depth_mode = seed.gl_ndc; // no geometry stage
        const std::bitset<32> ints = IntegerAttributes(text);
        for (u32 i = 0; i < 32; ++i) {
            if (ints[i]) rt.generic_input_types[i] = Shader::AttributeType::UnsignedInt;
        }
    }
    bool pinned = false;
    auto building_cfg = std::make_shared<bool>(true);
    std::vector<GraphicsPipeline::TypeRead> types; // as for compute stages above
    ShaderHooks hooks{
        .cbuf = [&pinned, building_cfg](u32 bank, u32 offset) {
            if (*building_cfg) pinned = true; // control flow from constant words: unknown here
            return SeedCbuf(bank, offset);
        },
        .texture_type = [&types, &type_of, s](u32 handle) {
            types.push_back({s, handle, type_of(handle)});
            return types.back().type;
        },
    };
    StageBuild b;
    std::vector<u32> spirv;
    TranslateStageInto(b, code, *program, s, rt, std::move(hooks), building_cfg, &spirv);
    if (pinned) return false;
    SeedStore(seed, StoreKey(code, *program, s, StageAssumptions(s, rt, std::string())), std::move(spirv), b.info, std::move(types));
    // the library as Build makes it (kPreRasterDynamic / kFragmentDynamic, layouts of the stage's own set)
    const auto& profile = ShaderProfile();
    const VkPipelineViewportDepthClipControlCreateInfoEXT clip_control{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, nullptr, VK_TRUE};
    const VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                     seed.gl_ndc && profile.support_native_ndc ? &clip_control : nullptr,
                                                     0, 1, nullptr, 1, nullptr};
    const VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0,
                                                        VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                                        VK_FRONT_FACE_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    const VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0,
                                                           VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 0.0f, nullptr, VK_FALSE, VK_FALSE};
    const VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VkStage(s),
                                                *b.module, "main", nullptr};
    std::vector<VkDescriptorSetLayout> sets(s == kVertex ? 2 : 1, VK_NULL_HANDLE);
    if (b.set_layout) sets[SetOf(s)] = *b.set_layout;
    const Vulkan::vk::PipelineLayout layout = PipelineLayout(sets);
    const VkGraphicsPipelineLibraryCreateInfoEXT info{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, nullptr,
        static_cast<VkGraphicsPipelineLibraryFlagsEXT>(s == kVertex ? VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT
                                                                    : VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT)};
    VkGraphicsPipelineCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &info,
        .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
        .stageCount = 1,
        .pStages = &stage,
        .pDynamicState = s == kVertex ? &kPreRasterDynamic : &kFragmentDynamic,
        .layout = *layout,
        .renderPass = pass,
        .subpass = 0,
    };
    if (s == kVertex) {
        ci.pViewportState = &viewport;
        ci.pRasterizationState = &raster;
    } else {
        ci.pDepthStencilState = &depth;
        ci.pMultisampleState = &multisample;
    }
    // into our pipeline cache (2026-10-24: NVIDIA's own, ~1 GB in all, evicted the seed; it was filled alone before)
    const auto t0 = std::chrono::steady_clock::now();
    const VkPipelineCache cache = DiskCache();
    PipelineTimer timer;
    timer.info.pNext = ci.pNext;
    if (timer.Next()) ci.pNext = timer.Next();
    const Vulkan::vk::Pipeline library = Vk().device->GetLogical().CreateGraphicsPipeline(ci, cache);
    seed.create_ns.fetch_add(static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
    if (timer.feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT &&
        timer.feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) {
        seed.cache_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

void SeedWorker(Seed& seed) {
    std::array<VkFormat, kMaxColorTargets> colors{};
    colors[0] = VK_FORMAT_R8G8B8A8_UNORM;
    const VkRenderPass pass = TargetsPass(colors, 1, VK_FORMAT_D32_SFLOAT_S8_UINT); // any: see above
    for (size_t i; (i = seed.next.fetch_add(1)) < seed.names.size();) {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            (SeedStage(seed, seed.names[i], pass) ? seed.built : seed.left).fetch_add(1);
        } catch (const std::exception& e) {
            if (seed.failed.fetch_add(1) < 5) Log("nvn seed: %s: %s", seed.names[i].c_str(), e.what());
        }
        seed.driver_ns.fetch_add(static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
        const size_t done = seed.built + seed.left + seed.failed;
        if (done % 2000 == 0) {
            Log("nvn seed: %zu of %zu stages (%.0f s)", done, seed.names.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - seed.start).count());
        }
    }
    if (seed.finished.fetch_add(1) + 1 == seed.threads) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - seed.start).count();
        Log("nvn seed: done in %.0f s: %zu stages built into our pipeline cache, %zu left to play (geometry, "
            "pinned control flow), %zu failed; %.0f s of work on %u threads, the driver's library creations %.2f ms "
            "per stage built; texture types asked %zu times, %zu found in the text (the rest: the stage's rule); %zu "
            "already in our pipeline cache; the marker is written with the cache's next write",
            s, seed.built.load(), seed.left.load(), seed.failed.load(), seed.driver_ns / 1e9, seed.threads,
            seed.built ? seed.create_ns / 1e6 / seed.built : 0.0, seed.types_from_text + seed.types_guessed,
            seed.types_from_text.load(), seed.cache_hits.load());
        Log("nvn seed: %zu stages into the shader store, %zu not (a texture handle of shifted, arrayed or more than two words)",
            seed.stored.load(), seed.unstorable.load());
        StoreFlush();
        Disk& d = TheDisk();
        std::lock_guard lock(d.mu);
        d.marker = seed.marker;
        d.marker_tag = seed.tag;
    }
}

/// Starts the seed in the background (once), or does nothing when RT_NVN_SEED is unset or the marker says it is done.
/// `wait`: return only when it is done (RT_NVN_SEED_ONLY).
void StartSeed(bool gl_ndc, bool wait) {
    static std::once_flag once;
    std::call_once(once, [&] {
        const char* dir = std::getenv("RT_NVN_SEED");
        if (!dir || !*dir) return;
        if (!GplOn()) {
            Log("nvn seed: needs pipelines linked from libraries (RT_NVN_GPL): not seeding");
            return;
        }
        auto* seed = new Seed; // lives until the process ends (its workers are detached)
        seed->dir = dir;
        seed->gl_ndc = gl_ndc;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(seed->dir, ec)) {
            const std::string n = e.path().filename().string();
            if (n.size() > 5 && n.ends_with(".ctrl")) seed->names.push_back(n.substr(0, n.size() - 5));
        }
        std::sort(seed->names.begin(), seed->names.end());
        u64 names_hash = 0xcbf29ce484222325ull; // which stages, not only how many
        for (const auto& n : seed->names) {
            for (const char c : n) names_hash = (names_hash ^ static_cast<u8>(c)) * 0x100000001b3ull;
        }
        const VkPhysicalDeviceProperties props = Vk().device->GetPhysical().GetProperties();
        char tag[256];
        std::snprintf(tag, sizeof tag, "seed 6; %s; driver 0x%x; %zu stages (%016llx); depth mode %d; ", props.deviceName,
                      props.driverVersion, seed->names.size(), (unsigned long long)names_hash, gl_ndc ? 0 : 1);
        seed->tag = std::string(tag) + (ShaderStoreOn() ? StoreTag() : std::string("no store")) + "\n";
        const char* cache_dir = std::getenv("RT_PIPELINE_CACHE");
        char name[48];
        std::snprintf(name, sizeof name, "%016llx.nvn.seed", gpu::ProgramId());
        seed->marker = std::filesystem::path(cache_dir && *cache_dir && std::string(cache_dir) != "0" ? cache_dir : "cache") / name;
        std::filesystem::create_directories(seed->marker.parent_path(), ec);
        DiskCache(); // (its path: the marker only counts with the file that holds the seed)
        if (std::ifstream in{seed->marker}; in && std::string(std::istreambuf_iterator<char>(in), {}) == seed->tag &&
                                              std::filesystem::exists(TheDisk().path, ec) && StoreFilled()) {
            Log("nvn seed: done before for this driver (%s)", seed->marker.string().c_str());
            return;
        }
        if (seed->names.empty()) {
            Log("nvn seed: no <hash>.ctrl stages in %s", dir);
            return;
        }
        const char* n = std::getenv("RT_NVN_SEED_THREADS");
        seed->threads = std::min(n ? static_cast<u32>(std::strtoul(n, nullptr, 10)) : 4u,
                                 std::max(1u, std::thread::hardware_concurrency() / 4));
        seed->start = std::chrono::steady_clock::now();
        Log("nvn seed: %zu stages from %s on %u threads%s", seed->names.size(), dir, seed->threads,
            wait ? "" : ", in the background");
        std::vector<std::thread> workers;
        for (u32 i = 0; i < seed->threads; ++i) workers.emplace_back([seed] { SeedWorker(*seed); });
        for (auto& w : workers) wait ? w.join() : w.detach();
    });
}

/// Fills `p` (the caller's, so that what the translation read survives a failure: BuildOrRemember).
void Build(GraphicsPipeline& p, const RenderState& rs, u32 mode, const TargetSet& targets, const ClipModes& clip) {
    const auto& device = *Vk().device;
    const auto& logical = device.GetLogical();
    const Shader::Profile& profile = ShaderProfile();

    // --- shaders, alone (BuildStagesAlone) or per pipeline
    const bool gl_ndc = clip.depth_mode == 0; // NVN_DEPTH_MODE_NEAR_IS_MINUS_W: clip z from -w to w
    std::vector<VkPipelineShaderStageCreateInfo> stage_infos;
    bool has_fragment = false;
    if (StageAloneOn()) {
        has_fragment = BuildStagesAlone(p, rs, mode, targets, gl_ndc, stage_infos);
    } else {
        // --- shaders (RT_NVN_STAGE_ALONE=0): translate every bound stage, then emit them in order with one binding
        // numbering
        std::vector<StageWork> work;
        for (u32 s : kPipelineOrder) {
            if (!rs.stages[s]) continue;
            if (s == kTessControl || s == kTessEvaluation) Unsupported("tessellation");
            StageWork w;
            w.nvn_stage = s;
            w.program = *rs.stages[s];
            w.building_cfg = std::make_shared<bool>(true);
            auto recorder = std::make_shared<TypeRecorder>();
            ShaderHooks hooks{
                .cbuf = [&p, &rs, s, cfg = w.building_cfg, recorder](u32 bank, u32 offset) {
                    const u32 v = CbufWord(rs, s, bank, offset);
                    if (*cfg) p.cbuf_reads.push_back({s, bank, offset, v}); // see ReadsThatPin
                    else recorder->Word(bank, offset, v);
                    return v;
                },
                .texture_type = [&p, &rs, s, recorder](u32 handle) {
                    p.type_reads.push_back(recorder->Type(rs, s, handle));
                    return p.type_reads.back().type;
                },
            };
            w.env = std::make_unique<NvnEnvironment>(ReadCode(w.program), w.program, std::move(hooks));
            w.flow_blocks = std::make_unique<Shader::ObjectPool<Shader::Maxwell::Flow::Block>>();
            w.insts = std::make_unique<Shader::ObjectPool<Shader::IR::Inst>>();
            w.blocks = std::make_unique<Shader::ObjectPool<Shader::IR::Block>>();
            const u32 cfg_offset = w.env->StartAddress() + static_cast<u32>(sizeof(Shader::ProgramHeader));
            Shader::Maxwell::Flow::CFG cfg(*w.env, *w.flow_blocks, cfg_offset, false);
            *w.building_cfg = false;
            w.ir = Shader::Maxwell::TranslateProgram(*w.insts, *w.blocks, *w.env, cfg, ShaderHostInfo());
            RescalingPass(w.ir);
            work.push_back(std::move(w));
        }
        if (work.empty() || work.front().nvn_stage != kVertex) Unsupported("a draw without a vertex program");
        // Without a fragment program (depth-only draws: shadow maps, UE4's occlusion queries) the colours are
        // undefined in Vulkan: no colour writes then (below).
        has_fragment = work.back().nvn_stage == kFragment;

        const bool has_geometry = rs.stages[kGeometry].has_value();
        Shader::Backend::Bindings bindings;
        Vulkan::DescriptorLayoutBuilder layout_builder{device};
        const Shader::IR::Program* previous = nullptr;
        for (StageWork& w : work) {
            Shader::RuntimeInfo rt;
            if (previous) {
                rt.previous_stage_stores = previous->info.stores;
                rt.previous_stage_legacy_stores_mapping = previous->info.legacy_stores_mapping;
            } else {
                rt.previous_stage_stores.mask.set();
            }
            switch (w.nvn_stage) {
            case kVertex:
                if (!has_geometry) rt.convert_depth_mode = gl_ndc;
                for (u32 i = 0; i < rs.vertex_attrib_count; ++i) {
                    const u32 f = rs.vertex_attribs[i] & 0xFF;
                    if (f) rt.generic_input_types[i] = FormatType(VertexFormat(f));
                }
                break;
            case kGeometry:
                rt.convert_depth_mode = gl_ndc;
                rt.input_topology = GeometryInput(mode);
                break;
            case kFragment: {
                const ColorState cs = std::bit_cast<ColorState>(rs.color);
                if (cs.alpha_test != 8) Unsupported("alpha test %u", cs.alpha_test); // needs the reference value
                for (u32 i = 0; i < targets.color_count; ++i) {
                    if (targets.colors[i] != VK_FORMAT_UNDEFINED) rt.color_output_types[i] = FormatType(targets.colors[i]);
                }
                break;
            }
            }
            if (StageDepsOn()) LogStageDeps(p, w.program, w.nvn_stage, rs, rt);
            Shader::Maxwell::ConvertLegacyToGeneric(w.ir, rt);
            DumpStage(w.program, w.ir);
            SetRescaleBase(bindings, w.ir.stage); // the stage's own bits, as when built alone (shaders.h RescaleRange)
            std::vector<u32> spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, rt, w.ir, bindings);
            if (w.nvn_stage != kFragment) MakePositionInvariant(spirv);
            p.modules.push_back(logical.CreateShaderModule({
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = spirv.size() * 4,
                .pCode = spirv.data(),
            }));
            layout_builder.Add(w.ir.info, VkStage(w.nvn_stage));
            p.uses_render_area |= w.ir.info.uses_render_area;
            stage_infos.push_back({
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VkStage(w.nvn_stage),
                .module = *p.modules.back(),
                .pName = "main",
                .pSpecializationInfo = nullptr,
            });
            p.stages.push_back({w.nvn_stage, w.program, w.ir.info});
            previous = &w.ir;
        }
        p.push = UsePush(layout_builder);
        p.set_layout = layout_builder.CreateDescriptorSetLayout(p.push);
        p.layout = layout_builder.CreatePipelineLayout(*p.set_layout);
        p.update = layout_builder.CreateTemplate(*p.set_layout, *p.layout, p.push);
    }

    // --- vertex input: the attributes the vertex shader loads, as the GPU fetches only those (a depth-only
    // pass keeps the mesh's whole layout bound with only its position stream: 20 draws of the plaza load)
    std::vector<VkVertexInputBindingDescription> vbindings;
    std::vector<VkVertexInputAttributeDescription> vattribs;
    const Shader::Info* vertex_info = nullptr;
    for (const GraphicsPipeline::Stage& st : p.stages) {
        if (st.nvn_stage == kVertex) vertex_info = &st.info;
    }
    u32 streams_used = 0;
    for (u32 i = 0; i < rs.vertex_attrib_count; ++i) {
        const u32 bits = rs.vertex_attribs[i];
        if (!(bits & 0xFF)) continue; // no format: not read
        if (vertex_info && !vertex_info->loads.Generic(i)) continue;
        const u32 stream = (bits >> 8) & 0x1F;
        if (stream >= rs.vertex_stream_count) Unsupported("attribute %u reads stream %u of %u", i, stream, rs.vertex_stream_count);
        const VkFormat f = VertexFormat(bits & 0xFF);
        if (f == VK_FORMAT_UNDEFINED) Unsupported("vertex format 0x%x", bits & 0xFF);
        vattribs.push_back({i, stream, SupportedVertexFormat(f), bits >> 13});
        streams_used |= 1u << stream;
    }
    for (u32 s = 0; s < rs.vertex_stream_count; ++s) {
        if (!(streams_used & (1u << s))) continue;
        const auto [stride, divisor] = rs.vertex_streams[s];
        if (divisor > 1) Unsupported("vertex stream divisor %u", divisor);
        vbindings.push_back({s, stride, divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
    }
    p.vertex_streams = streams_used;
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = static_cast<u32>(vbindings.size()),
        .pVertexBindingDescriptions = vbindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(vattribs.size()),
        .pVertexAttributeDescriptions = vattribs.data(),
    };
    const VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = Topology(mode),
        .primitiveRestartEnable = VK_FALSE,
    };

    // --- viewport (dynamic), with NVN's -1..1 clip depth where the device has depth clip control
    const VkPipelineViewportDepthClipControlCreateInfoEXT clip_control{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .pNext = nullptr,
        .negativeOneToOne = VK_TRUE,
    };
    const VkPipelineViewportStateCreateInfo viewport{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = gl_ndc && profile.support_native_ndc ? &clip_control : nullptr,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = nullptr,
        .scissorCount = 1,
        .pScissors = nullptr,
    };

    // --- rasterization
    const PolygonState poly = std::bit_cast<PolygonState>(rs.polygon);
    if (poly.polygon_mode > 2 || poly.cull_face > 3 || poly.front_face > 1) Unsupported("polygon state");
    const u32 offset_bit = poly.polygon_mode == 0 ? 1 : poly.polygon_mode == 1 ? 2 : 4;
    const bool bias = (poly.offset_enables & offset_bit) != 0;
    const VkPipelineRasterizationStateCreateInfo raster{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = poly.polygon_mode == 0 ? VK_POLYGON_MODE_POINT : poly.polygon_mode == 1 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL,
        .cullMode = static_cast<VkCullModeFlags>(poly.cull_face), // 1 front, 2 back, 3 both: Vulkan's bits
        .frontFace = poly.front_face ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = bias ? VK_TRUE : VK_FALSE,
        .depthBiasConstantFactor = rs.polygon_offset[1],
        .depthBiasClamp = rs.polygon_offset[2],
        .depthBiasSlopeFactor = rs.polygon_offset[0],
        .lineWidth = 1.0f,
    };

    // --- multisample
    if (rs.multisample_set) {
        const MultisampleState ms = std::bit_cast<MultisampleState>(rs.multisample);
        if (ms.samples > 1 || ms.raster_samples > 1) Unsupported("multisampling (%u samples)", ms.samples);
    }
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    // --- depth and stencil (only with a depth target)
    VkPipelineDepthStencilStateCreateInfo depth{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = VK_FALSE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {},
        .back = {},
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };
    if (targets.depth != VK_FORMAT_UNDEFINED) {
        const DepthStencilState ds = std::bit_cast<DepthStencilState>(rs.depth_stencil);
        depth.depthTestEnable = ds.flags & 1 ? VK_TRUE : VK_FALSE;
        depth.depthWriteEnable = ds.flags & 2 ? VK_TRUE : VK_FALSE;
        depth.depthCompareOp = CompareOp(ds.depth_func);
        depth.stencilTestEnable = ds.flags & 4 ? VK_TRUE : VK_FALSE;
        depth.front = StencilState(ds.stencil_funcs & 15, ds.front_ops, rs, 0);
        depth.back = StencilState(ds.stencil_funcs >> 4, ds.back_ops, rs, 1);
        depth.depthBoundsTestEnable = rs.depth_bounds_enable ? VK_TRUE : VK_FALSE;
    }

    // --- colour blending, per target
    const ColorState cs = std::bit_cast<ColorState>(rs.color);
    if (cs.logic_op != 3) Unsupported("logic op %u", cs.logic_op); // 3: COPY
    const u32 masks = std::bit_cast<ChannelMaskState>(rs.channel_mask).masks;
    std::array<VkPipelineColorBlendAttachmentState, kMaxColorTargets> blends{};
    for (u32 i = 0; i < targets.color_count; ++i) {
        const BlendState b = std::bit_cast<BlendState>(rs.blend[i]);
        const bool on = (cs.blend_enable >> i) & 1;
        if (on && b.advanced_mode != 0) Unsupported("advanced blend mode %u", b.advanced_mode);
        blends[i] = {
            .blendEnable = on ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = on ? BlendFactor(b.src) : VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = on ? BlendFactor(b.dst) : VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = on ? BlendOp(b.equations & 15) : VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = on ? BlendFactor(b.src_alpha) : VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = on ? BlendFactor(b.dst_alpha) : VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = on ? BlendOp(b.equations >> 4) : VK_BLEND_OP_ADD,
            .colorWriteMask = has_fragment ? (masks >> (4 * i)) & 15 : 0u, // NVN r g b a bits = Vulkan's
        };
    }
    const VkPipelineColorBlendStateCreateInfo blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = targets.color_count,
        .pAttachments = blends.data(),
        .blendConstants = {},
    };

    std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                       VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    if (depth.depthBoundsTestEnable) dyn.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
    const VkPipelineDynamicStateCreateInfo dynamic{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(dyn.size()),
        .pDynamicStates = dyn.data(),
    };
    PipelineTimer timer;
    if (GplOn()) {
        // linked from four libraries (LibraryFor): vertex input, pre-rasterisation shaders (the vertex and geometry
        // stage builds), fragment shader (the fragment stage build), fragment output; the state the shader
        // libraries would hold is set at each draw (p.dynamic, draws.cpp), so one library serves a stage build
        // in every pipeline of its render pass (libraries of one pipeline must have compatible render passes:
        // ours are one per format set, render.cpp TargetsPass)
        const auto built = [&](u32 s) -> const StageBuild* {
            for (const auto& st : p.stages) {
                if (st.nvn_stage == s) return st.built.get();
            }
            return nullptr;
        };
        const StageBuild* const vs = built(kVertex);
        const StageBuild* const gs = built(kGeometry);
        const StageBuild* const fs = built(kFragment);
        bool hit = true;
        std::string key = "vertex input";
        PutBytes(key, input_assembly.topology);
        for (const auto& b : vbindings) PutBytes(key, b);
        key += '|';
        for (const auto& a : vattribs) PutBytes(key, a);
        p.libraries.push_back(LibraryFor(VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, key, VK_NULL_HANDLE,
                                         {}, {}, hit, [&](VkGraphicsPipelineCreateInfo& ci) {
            ci.pVertexInputState = &vertex_input;
            ci.pInputAssemblyState = &input_assembly;
        }));
        key = "pre-rasterisation";
        PutBytes(key, vs);
        PutBytes(key, gs);
        PutBytes(key, targets.pass);
        PutBytes(key, raster.polygonMode);
        PutBytes(key, viewport.pNext != nullptr);
        std::vector<VkDescriptorSetLayout> pre_sets(gs ? 3 : 2, VK_NULL_HANDLE);
        if (vs->set_layout) pre_sets[SetOf(kVertex)] = *vs->set_layout;
        if (gs && gs->set_layout) pre_sets[SetOf(kGeometry)] = *gs->set_layout;
        const u32 pre_raster = static_cast<u32>(stage_infos.size()) - (has_fragment ? 1 : 0);
        p.libraries.push_back(LibraryFor(VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, key,
                                         targets.pass, pre_sets, kPreRasterDynamic, hit, [&](VkGraphicsPipelineCreateInfo& ci) {
            ci.stageCount = pre_raster;
            ci.pStages = stage_infos.data();
            ci.pViewportState = &viewport;
            ci.pRasterizationState = &raster;
        }));
        key = "fragment shader";
        PutBytes(key, fs);
        PutBytes(key, targets.pass);
        std::vector<VkDescriptorSetLayout> fs_sets(1, VK_NULL_HANDLE);
        if (fs && fs->set_layout) fs_sets[0] = *fs->set_layout;
        p.libraries.push_back(LibraryFor(VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, key, targets.pass,
                                         fs_sets, kFragmentDynamic, hit, [&](VkGraphicsPipelineCreateInfo& ci) {
            ci.stageCount = has_fragment ? 1 : 0;
            ci.pStages = has_fragment ? &stage_infos.back() : nullptr;
            ci.pDepthStencilState = &depth;
            ci.pMultisampleState = &multisample;
        }));
        key = "fragment output";
        PutBytes(key, targets.pass);
        PutBytes(key, targets.color_count);
        for (u32 i = 0; i < targets.color_count; ++i) PutBytes(key, blends[i]);
        p.libraries.push_back(LibraryFor(VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, key,
                                         targets.pass, {}, kOutputDynamic, hit, [&](VkGraphicsPipelineCreateInfo& ci) {
            ci.pColorBlendState = &blend;
            ci.pMultisampleState = &multisample;
        }));
        NoteLibraryPass(vs, targets.pass);
        NoteLibraryPass(fs, targets.pass);
        p.pipeline = Link(p, targets.pass, false);
        p.linked = true;
        p.dynamic = {
            .cull = raster.cullMode,
            .front = raster.frontFace,
            .bias = raster.depthBiasEnable == VK_TRUE,
            .bias_constant = raster.depthBiasConstantFactor,
            .bias_clamp = raster.depthBiasClamp,
            .bias_slope = raster.depthBiasSlopeFactor,
            .depth_test = depth.depthTestEnable == VK_TRUE,
            .depth_write = depth.depthWriteEnable == VK_TRUE,
            .depth_bounds = depth.depthBoundsTestEnable == VK_TRUE,
            .stencil_test = depth.stencilTestEnable == VK_TRUE,
            .depth_compare = depth.depthCompareOp,
            .front_ops = depth.front,
            .back_ops = depth.back,
        };
        t_cache_hit = hit ? 1 : 0; // whether the driver's cache had every new library (the link itself is not cached)
        if (OptimiseOn()) TheOptimiser().Add(p, targets.pass);
        return;
    }
    p.pipeline = logical.CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = timer.Next(),
        .flags = 0,
        .stageCount = static_cast<u32>(stage_infos.size()),
        .pStages = stage_infos.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = nullptr,
        .pViewportState = &viewport,
        .pRasterizationState = &raster,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic,
        .layout = *p.layout,
        .renderPass = targets.pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    }, DiskCache());
}

} // namespace

VkFormat VertexFormat(u32 id) {
    // LASRecomp's vertex_format (runtime/nvn/vulkan/exec.c, MIT): NVN's colour formats 0x01-0x30 read as
    // vertex data, and the packed ones.
    constexpr VkFormat k[0x31] = {
        VK_FORMAT_UNDEFINED,
        VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT,
        VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
        VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT,
        VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT,
        VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT,
        VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT,
        VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16_SINT,
        VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT,
        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT,
        VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
    };
    if (id < std::size(k)) return k[id];
    switch (id) {
    case 0x3d: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 0x3f: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case 0x40: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case 0x69: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    default: return VK_FORMAT_UNDEFINED;
    }
}

std::array<u8, kComputeC0Size> ComputeC0(const RenderState& rs) {
    std::array<u8, kComputeC0Size> c0{};
    std::memcpy(c0.data() + kComputeC0Groups, rs.dispatch.data(), 12);
    const auto& b = rs.bindings[5];
    for (u32 i = 0; i < 16; ++i) {
        const auto put = [&](u32 base, const std::array<u64, 2>& r) {
            const u32 e[3] = {static_cast<u32>(r[0]), static_cast<u32>(r[0] >> 32), static_cast<u32>(r[1])};
            std::memcpy(c0.data() + base + 16 * i, e, 12);
        };
        put(kComputeC0Uniform, b.uniform_buffers[i]);
        put(kComputeC0Storage, b.storage_buffers[i]);
    }
    return c0;
}

u32 CbufWord(const RenderState& rs, u32 nvn_stage, u32 bank, u32 offset) {
    u32 v = 0;
    if (bank == 0 && nvn_stage == 5) {
        if (offset + 4 <= kComputeC0Size) std::memcpy(&v, ComputeC0(rs).data() + offset, 4);
    } else if (bank == 0) {
        if (offset + 4 <= kC0Size) std::memcpy(&v, rs.c0.data() + offset, 4);
    } else if (bank == 2) {
        if (offset + 4 <= kC2Size) std::memcpy(&v, rs.c2[C2Index(nvn_stage)].data() + offset, 4);
    } else if (bank >= 3 && bank - 3 < rs.bindings[nvn_stage].uniform_buffers.size()) {
        const auto [address, size] = rs.bindings[nvn_stage].uniform_buffers[bank - 3];
        if (address && offset + 4 <= size) v = hle::ReadGuest<u32>(address + offset);
    } else {
        Fatal("nvn: constant buffer c%u read at translation", bank);
    }
    return v;
}

Shader::TextureType TextureTypeOf(const RenderState& rs, u32 handle, bool* registered) {
    const auto e = TexturePoolEntry(rs.texture_pool, handle & 0xFFFFF);
    if (registered) *registered = e.has_value();
    if (!e) return Shader::TextureType::Color2D;
    u32 target = e->texture.settings.target;
    if (e->view.present && (e->view.view.set & kViewTarget)) target = e->view.view.target;
    switch (target) { // NVNtextureTarget
    case 0: return Shader::TextureType::Color1D;
    case 1:
    case 5: return Shader::TextureType::Color2D; // 2D, 2D multisample
    case 2: return Shader::TextureType::Color3D;
    case 3: return Shader::TextureType::ColorArray1D;
    case 4:
    case 6: return Shader::TextureType::ColorArray2D;
    case 7: return Shader::TextureType::Color2DRect;
    case 8: return Shader::TextureType::ColorCube;
    case 9: return Shader::TextureType::ColorArrayCube;
    case 10: return Shader::TextureType::Buffer;
    }
    Fatal("nvn: texture target %u", target);
}

namespace {

// RT_NVN_PIPELINE_THREADS=<n> (SHADER_PLAN Phase 3; default 4, at most a quarter of the host's threads; 0 or
// RT_NVN_SYNC_PIPELINES=1: built on the draw's thread, as before): new graphics pipelines are built on n worker
// threads from a copy of the draw's state. RT_NVN_PIPELINE_WAIT=<ms> (default -1: always wait): how long the
// draws of one frame (present to present) may wait for their pipelines in all before a draw whose pipeline is
// not built is skipped (NotReady) - a never-seen pipeline then shows a few frames late instead of holding the
// frame for its compile; one whose driver cache hits is usually done within the wait. A budget per frame, not per
// draw: in the user's play run a new area brought 82 new pipelines in one frame (2026-10-23, 114 ms translating).
// Always waiting is the default since a skipped draw that the game renders only once stayed missing for good (the
// plaza washed out to white: SHADER_PLAN Phase 3); first-time compiles are for the seed to remove (Phase 5).
// Translation stays serialised (StageFor's lock: drippu's translator is not known to be re-entrant); the driver's
// library compiles run in parallel (LibraryFor). Compute pipelines are still built on the dispatch's thread (a
// skipped dispatch can leave data later frames read).
std::atomic<s64> g_wait_left_ns{0}; // what the frame's draws may still wait for their pipelines (PipelineFor)

struct PipelineBuilds {
    u32 threads = 0;
    int wait_ms = -1;
};
const PipelineBuilds& Async() {
    static const PipelineBuilds a = [] {
        PipelineBuilds a;
        const char* sync = std::getenv("RT_NVN_SYNC_PIPELINES");
        const char* n = std::getenv("RT_NVN_PIPELINE_THREADS");
        const char* wait = std::getenv("RT_NVN_PIPELINE_WAIT");
        a.threads = std::min(n ? static_cast<u32>(std::strtoul(n, nullptr, 10)) : 4u,
                             std::max(1u, std::thread::hardware_concurrency() / 4));
        if (sync && std::strcmp(sync, "0") != 0) a.threads = 0;
        if (wait) a.wait_ms = std::atoi(wait);
        if (a.threads) {
            Log("nvn: graphics pipelines built on %u worker threads; a draw waits %s for its pipeline", a.threads,
                a.wait_ms < 0 ? "until it is built" : (std::to_string(a.wait_ms) + " ms at most, then is skipped").c_str());
        } else {
            Log("nvn: graphics pipelines built on the draw's thread");
        }
        return a;
    }();
    return a;
}

class Builder {
public:
    struct Job {
        GraphicsPipeline* p; // in Pipelines(), building
        std::unique_ptr<RenderState> rs;
        u32 mode;
        TargetSet targets;
        ClipModes clip;
    };
    void Add(Job job) {
        std::lock_guard lock(mu_);
        while (workers_ < Async().threads) {
            std::thread([this] { Run(); }).detach(); // run until the process ends
            ++workers_;
        }
        jobs_.push_back(std::move(job));
        cv_.notify_one();
    }

private:
    void Run() {
        for (;;) {
            Job job;
            {
                std::unique_lock lock(mu_);
                cv_.wait(lock, [&] { return !jobs_.empty(); });
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            try {
                BuildLogged(*job.p, "graphics", [&](GraphicsPipeline& p) { Build(p, *job.rs, job.mode, job.targets, job.clip); });
            } catch (const std::exception& e) { // not NotYet (BuildLogged keeps those): e.g. drippu's translator
                job.p->failed = e.what();
                Log("nvn: building a pipeline failed: %s", e.what());
            }
            Cache& c = Pipelines();
            {
                std::lock_guard lock(c.mu);
                job.p->building = false;
            }
            c.built.notify_all();
        }
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> jobs_;
    u32 workers_ = 0;
};
Builder& TheBuilder() {
    static Builder b;
    return b;
}

/// Four words of a stage's code memory - its first instructions, the middle and the end of the program, the
/// first compiler constants: cheap enough for every draw, and different for different code at one address in
/// all but contrived cases (LASRecomp's stale-program detector does the same).
std::array<u64, 4> Fingerprint(const ProgramStage& s) {
    const u64 program = s.code + s.program_offset;
    const u64 last = s.program_size >= 8 ? s.program_size - 8 : 0;
    return {hle::ReadGuest<u64>(program + std::min<u64>(0x50, last)), hle::ReadGuest<u64>(program + (last / 2 & ~u64{7})),
            hle::ReadGuest<u64>(program + last), s.constants_size >= 8 ? hle::ReadGuest<u64>(s.code + s.constants_offset) : 0};
}

/// Whether the stages bound in `rs` still have the code `p` was built from (logs the first few that don't).
bool SameCode(const GraphicsPipeline& p, const RenderState& rs) {
    for (size_t i = 0; i < p.stages.size() && i < p.fingerprints.size(); ++i) {
        if (Fingerprint(*rs.stages[p.stages[i].nvn_stage]) == p.fingerprints[i]) continue;
        static std::atomic<u32> told{0};
        if (told.fetch_add(1) < 10) {
            Log("nvn: other code at a pipeline's stage %u address 0x%llx: built anew", p.stages[i].nvn_stage,
                (unsigned long long)rs.stages[p.stages[i].nvn_stage]->code);
        }
        return false;
    }
    return true;
}

} // namespace

void PipelineFrameStart() { g_wait_left_ns.store(s64{std::max(Async().wait_ms, 0)} * 1'000'000, std::memory_order_relaxed); }

void RunSeedOnly() {
    const char* mode = std::getenv("RT_NVN_SEED_DEPTH_MODE");
    StartSeed(mode && std::strcmp(mode, "0") == 0, true);
    Exit(0);
}

const GraphicsPipeline& PipelineFor(const RenderState& rs, u32 mode, const TargetSet& targets) {
    const ClipModes clip = DeviceClipModes(rs.device);
    StartSeed(clip.depth_mode == 0, false); // RT_NVN_SEED: once, at the first pipeline (the device's depth mode is set)
    std::vector<u8> key;
    Put(key, clip);
    Put(key, mode);
    for (u32 s = 0; s < kStages - 1; ++s) {
        const auto& st = rs.stages[s];
        Put(key, st ? st->code : 0);
        if (!st) continue;
        Put(key, std::array<u32, 4>{st->program_offset, st->program_size, st->constants_offset, st->constants_size});
        Put(key, st->tag_word);
    }
    Put(key, rs.vertex_attrib_count);
    for (u32 i = 0; i < rs.vertex_attrib_count; ++i) Put(key, rs.vertex_attribs[i]);
    Put(key, rs.vertex_stream_count);
    for (u32 i = 0; i < rs.vertex_stream_count; ++i) Put(key, rs.vertex_streams[i]);
    Put(key, rs.polygon);
    Put(key, rs.polygon_offset);
    Put(key, rs.color);
    Put(key, rs.channel_mask);
    for (u32 i = 0; i < targets.color_count; ++i) Put(key, rs.blend[i]);
    Put(key, rs.multisample_set ? rs.multisample : std::array<u8, 24>{});
    if (targets.depth != VK_FORMAT_UNDEFINED) {
        Put(key, rs.depth_stencil);
        Put(key, rs.depth_bounds_enable);
    }
    Put(key, targets.pass);

    Cache& c = Pipelines();
    std::unique_lock lock(c.mu);
    auto& variants = c.pipelines[key];
    const PipelineBuilds& async = Async();
    // this draw may wait for what is left of the frame's budget (PipelineFrameStart); what it waited is spent
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::nanoseconds(std::max<s64>(g_wait_left_ns.load(std::memory_order_relaxed), 0));
    struct Spend {
        std::chrono::steady_clock::time_point start;
        bool waited = false;
        ~Spend() {
            if (waited) {
                g_wait_left_ns.fetch_sub(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - start).count(), std::memory_order_relaxed);
            }
        }
    } spend{start};
    for (;;) {
        GraphicsPipeline* building = nullptr;
        for (const auto& p : variants) {
            if (p->stale) continue;
            if (p->building) {
                building = p.get();
                continue;
            }
            if (!StillHolds(*p, rs)) continue;
            if (!SameCode(*p, rs)) { // other code at the same address
                p->stale = true;
                continue;
            }
            if (!p->failed.empty()) throw NotYet(p->failed);
            return *p;
        }
        if (!async.threads) {
            BuildOrRemember(variants, "graphics", [&](GraphicsPipeline& p) {
                Build(p, rs, mode, targets, clip);
                for (const auto& st : p.stages) p.fingerprints.push_back(Fingerprint(st.program));
            });
            return *variants.back();
        }
        if (!building) {
            auto fresh = std::make_unique<GraphicsPipeline>();
            fresh->building = true;
            for (u32 s : kPipelineOrder) {
                if (rs.stages[s]) fresh->fingerprints.push_back(Fingerprint(*rs.stages[s])); // of the code as bound now
            }
            building = fresh.get();
            variants.push_back(std::move(fresh));
            TheBuilder().Add({building, std::make_unique<RenderState>(rs), mode, targets, clip});
        }
        const auto done = [&] { return !building->building; };
        spend.waited = true;
        if (async.wait_ms < 0) {
            c.built.wait(lock, done);
        } else if (!c.built.wait_until(lock, deadline, done)) {
            TheTotals().not_ready.fetch_add(1, std::memory_order_relaxed);
            throw NotReady("its pipeline being built");
        }
        // built: look again (it may not hold for this draw's state, or have failed)
    }
}

namespace {

void BuildCompute(ComputePipeline& p, const RenderState& rs) {
    p.program = *rs.compute_stage;
    constexpr u32 s = 5;
    bool building_cfg = true;
    auto recorder = std::make_shared<TypeRecorder>();
    ShaderHooks hooks{
        .cbuf = [&p, &rs, &building_cfg, recorder](u32 bank, u32 offset) {
            const u32 v = CbufWord(rs, s, bank, offset);
            if (building_cfg) p.cbuf_reads.push_back({s, bank, offset, v}); // see ReadsThatPin
            else recorder->Word(bank, offset, v);
            return v;
        },
        .texture_type = [&p, &rs, recorder](u32 handle) {
            p.type_reads.push_back(recorder->Type(rs, s, handle));
            return p.type_reads.back().type;
        },
    };
    std::vector<u8> code = ReadCode(p.program);
    p.hash = CodeHash(code, p.program);
    // translated before play (the seed's, SHADER_PLAN Phase 4) and its checks hold: no translation
    std::optional<StoredStage> stored;
    if (ShaderStoreOn()) {
        stored = StoreGet(StoreKey(code, p.program, s, std::string()));
        if (stored && !StoredHolds(*stored, rs)) stored.reset();
        if (stored && !ShaderStoreCheck()) {
            p.info = stored->info;
            p.cbuf_reads = stored->cbuf_reads;
            p.type_reads = stored->type_reads;
            FinishCompute(p, stored->spirv, DiskCache());
            TheTotals().stages_stored.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    std::vector<u32> spirv;
    MakeCompute(p, std::move(code), std::move(hooks), building_cfg, DiskCache(), stored ? &spirv : nullptr);
    if (stored) CheckStored("compute", p.hash, *stored, spirv, p.info);
}

struct ComputeCache {
    std::mutex mu;
    std::map<std::vector<u8>, std::vector<std::unique_ptr<ComputePipeline>>> pipelines;
};

} // namespace

u64 StageContentHash(const ProgramStage& s) { return StageHash(s); }

const ComputePipeline& ComputePipelineFor(const RenderState& rs) {
    if (!rs.compute_stage) Unsupported("a dispatch without a compute program");
    const ProgramStage& st = *rs.compute_stage;
    std::vector<u8> key;
    Put(key, st.code);
    Put(key, std::array<u32, 8>{st.program_offset, st.program_size, st.constants_offset, st.constants_size,
                                st.workgroup[0], st.workgroup[1], st.workgroup[2], st.shared_memory});
    Put(key, st.local_memory);
    Put(key, st.tag_word);
    static ComputeCache c;
    std::lock_guard lock(c.mu);
    auto& variants = c.pipelines[key];
    for (const auto& p : variants) {
        if (!StillHolds(*p, rs)) continue;
        if (!p->failed.empty()) throw NotYet(p->failed);
        return *p;
    }
    BuildOrRemember(variants, "compute", [&](ComputePipeline& p) { BuildCompute(p, rs); });
    return *variants.back();
}

} // namespace rt::nvn
