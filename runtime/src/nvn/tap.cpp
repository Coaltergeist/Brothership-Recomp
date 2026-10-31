// The NVN tap: wrappers, census and argument summaries (tap.h).
#include "tap.h"

#include "../diag.h"
#include "../hle.h"
#include "../perf.h"
#include "../threads.h"
#include "layer.h"
#include "nvn.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace rt::nvn::tap {

namespace {

// --- Argument table ------------------------------------------------------------
// Which registers to record per function: x0..x7 as 0..7, v0..v7 as kV + i (raw
// bits). Meanings by NVN's public signatures, which agree with LASRecomp's
// bindings *(external)*; x0 is always the object.
constexpr int kV = 100;
struct ArgSpec {
    const char* name;
    std::array<int, 4> regs; // -1 = unused
    const char* labels;
};
const ArgSpec kArgSpecs[] = {
    {"nvnDeviceBuilderSetFlags", {1, -1, -1, -1}, "flags"},
    {"nvnDeviceGetInteger", {1, -1, -1, -1}, "which"},
    {"nvnQueueBuilderSetFlags", {1, -1, -1, -1}, "flags"},
    {"nvnQueueBuilderSetCommandMemorySize", {1, -1, -1, -1}, "size"},
    {"nvnQueueBuilderSetComputeMemorySize", {1, -1, -1, -1}, "size"},
    {"nvnQueueBuilderSetControlMemorySize", {1, -1, -1, -1}, "size"},
    {"nvnQueueBuilderSetCommandFlushThreshold", {1, -1, -1, -1}, "bytes"},
    {"nvnQueueBuilderSetQueueMemory", {2, -1, -1, -1}, "size"},
    {"nvnCommandBufferAddCommandMemory", {2, 3, -1, -1}, "offset size"},
    {"nvnCommandBufferAddControlMemory", {2, -1, -1, -1}, "size"},
    {"nvnMemoryPoolBuilderSetFlags", {1, -1, -1, -1}, "flags"},
    {"nvnMemoryPoolBuilderSetStorage", {2, -1, -1, -1}, "size"},
    {"nvnBufferBuilderSetStorage", {3, -1, -1, -1}, "size"},
    {"nvnTextureBuilderSetFlags", {1, -1, -1, -1}, "flags"},
    {"nvnTextureBuilderSetTarget", {1, -1, -1, -1}, "target"},
    {"nvnTextureBuilderSetFormat", {1, -1, -1, -1}, "format"},
    {"nvnTextureBuilderSetSize1D", {1, -1, -1, -1}, "w"},
    {"nvnTextureBuilderSetSize2D", {1, 2, -1, -1}, "w h"},
    {"nvnTextureBuilderSetSize3D", {1, 2, 3, -1}, "w h d"},
    {"nvnTextureBuilderSetLevels", {1, -1, -1, -1}, "levels"},
    {"nvnTextureBuilderSetSamples", {1, -1, -1, -1}, "samples"},
    {"nvnTextureBuilderSetStride", {1, -1, -1, -1}, "stride"},
    {"nvnTextureBuilderSetDepthStencilMode", {1, -1, -1, -1}, "mode"},
    {"nvnTextureViewSetFormat", {1, -1, -1, -1}, "format"},
    {"nvnTextureViewSetTarget", {1, -1, -1, -1}, "target"},
    {"nvnTexturePoolInitialize", {3, -1, -1, -1}, "descriptors"},
    {"nvnSamplerPoolInitialize", {3, -1, -1, -1}, "descriptors"},
    {"nvnSamplerBuilderSetMinMagFilter", {1, 2, -1, -1}, "min mag"},
    {"nvnSamplerBuilderSetWrapMode", {1, 2, 3, -1}, "s t r"},
    {"nvnSamplerBuilderSetCompare", {1, 2, -1, -1}, "mode func"},
    {"nvnSamplerBuilderSetReductionFilter", {1, -1, -1, -1}, "filter"},
    {"nvnSamplerBuilderSetMaxAnisotropy", {kV + 0, -1, -1, -1}, "aniso(f32 bits)"},
    {"nvnProgramSetShaders", {1, -1, -1, -1}, "count"},
    {"nvnWindowBuilderSetTextures", {1, -1, -1, -1}, "count"},
    {"nvnWindowBuilderSetPresentInterval", {1, -1, -1, -1}, "interval"},
    {"nvnWindowSetPresentInterval", {1, -1, -1, -1}, "interval"},
    {"nvnCommandBufferDrawArrays", {1, -1, -1, -1}, "mode"},
    {"nvnCommandBufferDrawArraysInstanced", {1, -1, -1, -1}, "mode"},
    {"nvnCommandBufferDrawElements", {1, 2, -1, -1}, "mode index_type"},
    {"nvnCommandBufferDrawElementsBaseVertex", {1, 2, -1, -1}, "mode index_type"},
    {"nvnCommandBufferDrawElementsInstanced", {1, 2, -1, -1}, "mode index_type"},
    {"nvnCommandBufferBindTexture", {1, 2, -1, -1}, "stage slot"},
    {"nvnCommandBufferBindTextures", {1, 2, 3, -1}, "stage first count"},
    {"nvnCommandBufferBindImage", {1, 2, -1, -1}, "stage slot"},
    {"nvnCommandBufferBindUniformBuffer", {1, 2, 4, -1}, "stage slot size"},
    {"nvnCommandBufferBindUniformBuffers", {1, 2, 3, -1}, "stage first count"},
    {"nvnCommandBufferBindStorageBuffer", {1, 2, 4, -1}, "stage slot size"},
    {"nvnCommandBufferBindStorageBuffers", {1, 2, 3, -1}, "stage first count"},
    {"nvnCommandBufferBindVertexBuffer", {1, -1, -1, -1}, "index"},
    {"nvnCommandBufferBindVertexBuffers", {1, 2, -1, -1}, "first count"},
    {"nvnCommandBufferBindVertexAttribState", {1, -1, -1, -1}, "count"},
    {"nvnCommandBufferBindVertexStreamState", {1, -1, -1, -1}, "count"},
    {"nvnCommandBufferSetRenderTargets", {1, -1, -1, -1}, "colors"},
    {"nvnCommandBufferDispatchCompute", {1, 2, 3, -1}, "x y z"},
    {"nvnCommandBufferBarrier", {1, -1, -1, -1}, "bits"},
    {"nvnCommandBufferClearColor", {1, 3, -1, -1}, "index mask"},
    {"nvnCommandBufferUpdateUniformBuffer", {4, -1, -1, -1}, "data_size"},
};

const ArgSpec* FindSpec(const std::string& name) {
    for (const ArgSpec& s : kArgSpecs) {
        if (name == s.name) return &s;
    }
    return nullptr;
}

// --- Tapped functions ------------------------------------------------------------

using Shadows = std::vector<ShadowFn>;

struct Fn {
    std::string name;
    u64 real = 0;      // the driver's function (lifted)
    u64 real_host = 0; // the wrapper's host address
    bool wraps_result = false; // nvnDeviceGetProcAddress: wrap what it returns
    const ArgSpec* spec = nullptr;
    std::atomic<const Shadows*> shadows{nullptr};

    std::mutex mu; // the counters below
    u64 order = 0; // rank of the first call
    u64 calls = 0, frames_with_calls = 0, max_per_frame = 0;
    u64 frame = ~0ULL, in_frame = 0; // the frame being counted, calls in it
    u64 first_frame = 0, last_frame = 0;
    double first_s = 0, last_s = 0;
    std::set<u64> threads; // KThread ids
    std::map<std::array<u64, 4>, u64> values; // recorded arguments -> calls
    u64 values_overflow = 0;
};

std::mutex g_mu; // g_fns, g_shadows
std::map<std::pair<std::string, u64>, std::unique_ptr<Fn>> g_fns; // (name, real) -> wrapper
std::map<std::string, const Shadows*> g_shadows; // lists are never freed: a call may be using one
std::atomic<u64> g_first_calls{0};
u64 g_bootstrap = 0; // nvnBootstrapLoader's guest address

constexpr size_t kMaxValues = 256; // distinct argument combinations kept per function

void Record(Fn* f, const GuestContext* c) {
    const u64 frame = perf::Frames();
    const double now = Uptime();
    const KThread* t = CurrentThread();
    std::lock_guard lock(f->mu);
    if (f->calls++ == 0) {
        f->order = ++g_first_calls;
        f->first_frame = frame;
        f->first_s = now;
    }
    f->last_frame = frame;
    f->last_s = now;
    if (frame != f->frame) {
        f->frame = frame;
        f->in_frame = 0;
        ++f->frames_with_calls;
    }
    if (++f->in_frame > f->max_per_frame) f->max_per_frame = f->in_frame;
    if (t) f->threads.insert(t->id);
    if (f->spec) {
        std::array<u64, 4> key{};
        for (int i = 0; i < 4; ++i) {
            const int r = f->spec->regs[i];
            key[i] = r < 0 ? 0 : r >= kV ? c->vreg[r - kV][0] : c->x[r];
        }
        auto it = f->values.find(key);
        if (it != f->values.end()) {
            ++it->second;
        } else if (f->values.size() < kMaxValues) {
            f->values.emplace(key, 1);
        } else {
            ++f->values_overflow;
        }
    }
}

u64 Wrap(const std::string& name, u64 real);

/// After the driver's nvnDeviceInitialize: every nvnDeviceGetInteger id with
/// the real device, for the layer's table of device-dependent values.
void ProbeDeviceInfo(GuestContext* c, u64 device) {
    auto lock = nvn::LockScratch();
    const u64 fn = nvn::RealProc(c, "nvnDeviceGetInteger");
    std::string line;
    for (u32 id = 0; id <= 105; ++id) {
        hle::WriteGuest<s32>(nvn::Scratch::kOut, 0);
        hle::CallGuest(c, fn, {device, id, nvn::Scratch::kOut}, true);
        char b[32];
        std::snprintf(b, sizeof b, " %u=0x%x", id, hle::ReadGuest<s32>(nvn::Scratch::kOut));
        line += b;
        if (id % 16 == 15 || id == 105) {
            Log("nvn tap: device info%s", line.c_str());
            line.clear();
        }
    }
}

void TapFn(GuestContext* c, void* user) {
    Fn* f = static_cast<Fn*>(user);
    Record(f, c);
    const Shadows* shadows = f->shadows.load(std::memory_order_acquire);
    std::unique_ptr<GuestContext> before;
    if (shadows) before = std::make_unique<GuestContext>(*c);
    std::string asked;
    if (f->wraps_result && c->x[1]) asked = hle::ReadGuestString(c->x[1], 256);
    const u64 object = c->x[0];
    hle::CallOriginal(c, f->real);
    if (f->wraps_result && c->x[0]) c->x[0] = Wrap(asked, c->x[0]);
    if (f->name == "nvnDeviceInitialize" && (c->x[0] & 0xFF)) ProbeDeviceInfo(c, object);
    if (shadows) for (ShadowFn fn : *shadows) fn(f->name, *before, *c);
}

/// The host address standing in for the driver's function `real` named `name`.
u64 Wrap(const std::string& name, u64 real) {
    if (hle::IsHostAddress(real)) return real; // already ours
    std::lock_guard lock(g_mu);
    auto& slot = g_fns[{name, real}];
    if (!slot) {
        slot = std::make_unique<Fn>();
        slot->name = name;
        slot->real = real;
        if (!FindFunc(real)) {
            Fatal("nvn tap: %s is at %s, which is not a lifted function start", name.c_str(), Describe(real).c_str());
        }
        slot->wraps_result = name == "nvnDeviceGetProcAddress";
        slot->spec = FindSpec(name);
        if (auto s = g_shadows.find(name); s != g_shadows.end()) slot->shadows.store(s->second);
        slot->real_host = hle::RegisterHostFunction("nvn tap: " + name, TapFn, slot.get());
    }
    return slot->real_host;
}

/// nvnBootstrapLoader(name): the driver's answer, wrapped.
void BootstrapTap(GuestContext* c, void*) {
    const std::string name = c->x[0] ? hle::ReadGuestString(c->x[0], 256) : "";
    hle::CallOriginal(c, g_bootstrap);
    Log("nvn tap: nvnBootstrapLoader(\"%s\") -> %s", name.c_str(), Describe(c->x[0]).c_str());
    if (c->x[0]) c->x[0] = Wrap(name, c->x[0]);
}

// --- Output ------------------------------------------------------------------------

std::string ThreadNames(const std::set<u64>& ids) {
    std::map<u64, std::string> names;
    for (const auto& t : ThreadsSnapshot()) {
        const std::string os = diag::OsThreadName(*t);
        names[t->id] = os.empty() ? t->name : os;
    }
    std::string out;
    for (u64 id : ids) {
        if (!out.empty()) out += ',';
        auto it = names.find(id);
        out += it != names.end() ? it->second : "thread " + std::to_string(id);
    }
    return out;
}

void WriteFile(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp";
    if (FILE* f = std::fopen(tmp.c_str(), "w")) {
        std::fwrite(text.data(), 1, text.size(), f);
        std::fclose(f);
        std::rename(tmp.c_str(), path.c_str());
    }
}

void WriteOutput() {
    struct Row {
        u64 order;
        std::string line;
    };
    std::vector<Row> census;
    std::string args = "name\targuments\tvalues\tcalls\n";
    std::vector<Fn*> fns;
    {
        std::lock_guard lock(g_mu);
        for (auto& [key, f] : g_fns) fns.push_back(f.get());
    }
    for (Fn* f : fns) {
        std::set<u64> threads;
        char b[512];
        {
            std::lock_guard lock(f->mu);
            if (!f->calls) continue;
            threads = f->threads;
            std::snprintf(b, sizeof b, "%llu\t%s\t%llu\t%llu\t%.2f\t%llu\t%llu\t%llu\t%.1f\t%.1f\t",
                          (unsigned long long)f->order, f->name.c_str(), (unsigned long long)f->calls,
                          (unsigned long long)f->frames_with_calls,
                          static_cast<double>(f->calls) / static_cast<double>(f->frames_with_calls),
                          (unsigned long long)f->max_per_frame, (unsigned long long)f->first_frame,
                          (unsigned long long)f->last_frame, f->first_s, f->last_s);
            if (f->spec) {
                std::vector<std::pair<u64, std::array<u64, 4>>> v;
                for (const auto& [k, n] : f->values) v.push_back({n, k});
                std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                for (const auto& [n, k] : v) {
                    std::string vals;
                    for (int i = 0; i < 4 && f->spec->regs[i] >= 0; ++i) {
                        char h[24];
                        std::snprintf(h, sizeof h, "%s0x%llx", i ? " " : "", (unsigned long long)k[i]);
                        vals += h;
                    }
                    args += f->name + "\t" + f->spec->labels + "\t" + vals + "\t" + std::to_string(n) + "\n";
                }
                if (f->values_overflow) {
                    args += f->name + "\t" + f->spec->labels + "\t(other)\t" + std::to_string(f->values_overflow) + "\n";
                }
            }
        }
        census.push_back({f->order, b + ThreadNames(threads)});
    }
    std::sort(census.begin(), census.end(), [](const Row& a, const Row& b) { return a.order < b.order; });
    char head[256];
    std::snprintf(head, sizeof head,
                  "# NVN tap: %zu functions called, %llu frames presented, t=%.0f s\n"
                  "order\tname\tcalls\tframes\tper_frame_mean\tper_frame_max\tfirst_frame\tlast_frame\tfirst_s\tlast_s\t"
                  "threads\n",
                  census.size(), (unsigned long long)perf::Frames(), Uptime());
    std::string text = head;
    for (const Row& r : census) text += r.line + "\n";
    WriteFile("nvn_census.tsv", text);
    WriteFile("nvn_args.tsv", args);
}

} // namespace

void AddShadow(const std::string& name, ShadowFn fn) {
    std::lock_guard lock(g_mu);
    const Shadows*& list = g_shadows[name];
    auto* grown = list ? new Shadows(*list) : new Shadows;
    grown->push_back(fn);
    list = grown;
    for (auto& [key, f] : g_fns) {
        if (key.first == name) f->shadows.store(list, std::memory_order_release);
    }
}

void Init() {
    if (!std::getenv("RT_NVN_TAP")) return;
    g_bootstrap = hle::OverrideGuestFunction("nvnBootstrapLoader", BootstrapTap);
    InstallShadows();
    Log("nvn tap: on (nvn_census.tsv, nvn_args.tsv every 10 s)");
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            WriteOutput();
        }
    }).detach();
}

} // namespace rt::nvn::tap
