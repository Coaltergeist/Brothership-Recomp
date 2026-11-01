// The NVN layer: the switch, the function table, the bootstrap, object headers
// and calls into the game's own SDK (layer.h, nvn.h).
#include "layer.h"

#include "../threads.h"
#include "nvn.h"
#include "shaders.h"
#include "tap.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace rt::nvn {

namespace {

const char* const kNames[] = {
#include "names.inc"
};

std::map<std::string, hle::HostFn>& Registry() {
    static std::map<std::string, hle::HostFn> r; // filled by static initialisers (NVN_PROC)
    return r;
}

struct Proc {
    const char* name;
    hle::HostFn fn; // the layer's, or Unimplemented
    std::atomic<bool> called{false};
};
std::unique_ptr<Proc[]> g_procs;
std::map<std::string, u64> g_addresses; // name -> host address; read-only after Init
std::atomic<u32> g_first_calls{0};
u64 g_bootstrap = 0; // the game's nvnBootstrapLoader

/// Every call into the layer passes here: logs a function's first call.
void Trampoline(GuestContext* c, void* user) {
    auto* p = static_cast<Proc*>(user);
    if (!p->called.load(std::memory_order_relaxed) && !p->called.exchange(true)) {
        Log("nvn: first call #%u %s", g_first_calls.fetch_add(1) + 1, p->name);
    }
    p->fn(c, p); // typed functions (hle::Bind) ignore it
}

void Unimplemented(GuestContext* c, void* user) {
    Fatal("nvn: %s is not implemented yet (NVN layer, RT_GPU=nvn), called from %s", static_cast<Proc*>(user)->name,
          Describe(c->x[30]).c_str());
}

/// nvnBootstrapLoader(name): the layer's function for `name`.
void Bootstrap(GuestContext* c, void*) {
    static std::once_flag difftest;
    if (std::getenv("RT_NVN_DIFFTEST")) std::call_once(difftest, [c] { RunDiffTest(c); });
    if (const char* dir = std::getenv("RT_NVN_SHADERTEST")) RunShaderTest(dir);
    if (std::getenv("RT_NVN_SEED_ONLY")) RunSeedOnly();
    const std::string name = c->x[0] ? hle::ReadGuestString(c->x[0], 256) : "";
    hle::RetX(c, Lookup(name));
    Log("nvn: nvnBootstrapLoader(\"%s\") -> %s", name.c_str(), Describe(c->x[0]).c_str());
}

std::set<std::pair<u32, s32>> g_shadow_seen; // (which, real value) logged once
std::mutex g_shadow_mu;

/// Tap mode: the real driver answered nvnDeviceGetInteger(device, which, out);
/// does the SDK with a stand-in device say the same?
void ShadowGetInteger(const std::string&, const GuestContext& before, const GuestContext&) {
    const u32 which = static_cast<u32>(before.x[1]);
    const s32 real = hle::ReadGuest<s32>(before.x[2]);
    {
        std::lock_guard lock(g_shadow_mu);
        if (!g_shadow_seen.insert({which, real}).second) return;
    }
    const s32 ours = DeviceInteger(CurrentContext(), which);
    Log("nvn shadow: nvnDeviceGetInteger(%u): driver 0x%x, layer 0x%x%s", which, real, ours,
        real == ours ? "" : "  DIFFERENT");
}

/// Tap mode: the real driver's queue start-up commands (its GPU macro upload,
/// constant-buffer bindings ...), saved for study as nvn_queue_init.bin in the
/// working directory: the words of the queue's internal command memory
/// (SDK 12.3.2: an NVNcommandBuffer at queue + 0x16d0; its chunk start at +48,
/// current position at +0). A research aid that reads NVN's internal layout,
/// so tap mode only.
void ShadowQueueInit(const std::string&, const GuestContext& before, const GuestContext& after) {
    if (!(after.x[0] & 0xFF)) return;
    const u64 cb = before.x[0] + 0x16d0;
    // The chunk start moves on at each flush: save from the start of its memory
    // region (at most 1 MB back) up to the current position.
    const u64 chunk = hle::ReadGuest<u64>(cb + 48), cur = hle::ReadGuest<u64>(cb);
    const Region* r = g_mem.Find(chunk);
    const u64 start = r ? std::max(r->base, chunk > (1u << 20) ? chunk - (1u << 20) : 0) : chunk;
    if (cur <= start || cur - start > (16u << 20)) {
        Log("nvn tap: queue start-up commands: unexpected range 0x%llx..0x%llx", (unsigned long long)start,
            (unsigned long long)cur);
        return;
    }
    std::vector<u8> words(cur - start);
    g_mem.Read(start, words.data(), words.size());
    if (FILE* f = std::fopen("nvn_queue_init.bin", "wb")) {
        std::fwrite(words.data(), 1, words.size(), f);
        std::fclose(f);
    }
    Log("nvn tap: queue start-up commands: %zu bytes from 0x%llx saved to nvn_queue_init.bin", words.size(),
        (unsigned long long)start);
}

} // namespace

bool Enabled() {
    static const bool on = [] {
        // auto (the default): the layer for a program that loads NVN (nvnBootstrapLoader in symbols.txt), the
        // driver seam for one that draws through nvdrv itself (M1's homebrew) and under the tap (RT_NVN_TAP: it
        // watches the real driver). After LoadSymbols; with no symbols at all (a player start, the game loaded at
        // Play, build.h) the layer: the start screen draws with it.
        const char* g = std::getenv("RT_GPU");
        const std::string v = g ? g : "auto";
        if (v != "auto" && v != "driver" && v != "nvn") Fatal("RT_GPU=%s: expected auto, driver or nvn", v.c_str());
        const bool nvn = v == "nvn" || (v == "auto" && !std::getenv("RT_NVN_TAP") &&
                                        (!HaveSymbols() || FindSymbol("nvnBootstrapLoader").size() == 1));
        Log("gpu: %s (RT_GPU=%s)", nvn ? "the NVN layer" : "the driver seam (video_core)", v.c_str());
        return nvn;
    }();
    return on;
}

void RegisterProc(const char* name, hle::HostFn fn) {
    if (!Registry().emplace(name, fn).second) Fatal("nvn: %s registered twice", name);
}

void Init() {
    if (!Enabled()) return;
    std::set<std::string> known(std::begin(kNames), std::end(kNames));
    for (const auto& [name, fn] : Registry()) {
        if (!known.count(name)) Fatal("nvn: the layer defines %s, which is no NVN name (names.inc)", name.c_str());
    }
    const size_t n = std::size(kNames);
    g_procs = std::make_unique<Proc[]>(n);
    for (size_t i = 0; i < n; ++i) {
        Proc& p = g_procs[i];
        p.name = kNames[i];
        auto it = Registry().find(p.name);
        p.fn = it != Registry().end() ? it->second : Unimplemented;
        g_addresses[p.name] = hle::RegisterHostFunction(p.name, Trampoline, &p);
    }
    g_bootstrap = hle::OverrideGuestFunction("nvnBootstrapLoader", Bootstrap);
    Log("nvn: NVN layer on (RT_GPU=nvn): %zu of %zu functions implemented", Registry().size(), n);
    std::string missing; // a call to one of these stops the run (Unimplemented)
    for (size_t i = 0; i < n; ++i) {
        if (!Registry().count(kNames[i])) missing += std::string(" ") + kNames[i];
    }
    Log("nvn: not implemented:%s", missing.c_str());
}

u64 Lookup(const std::string& name) {
    auto it = g_addresses.find(name);
    if (it != g_addresses.end()) return it->second;
    static std::mutex mu;
    static std::set<std::string> told;
    std::lock_guard lock(mu);
    if (told.insert(name).second) Log("nvn: no function %s (answered 0, as the driver does)", name.c_str());
    return 0;
}

const char* TypeName(Type t) {
    static const char* const names[] = {"?", "device", "queue", "commandBuffer", "memoryPool", "buffer", "texture",
                                        "sampler", "texturePool", "samplerPool", "program", "sync", "event", "window"};
    const u32 i = static_cast<u32>(t);
    return i < std::size(names) ? names[i] : "?";
}

Object* ObjectOf(u64 guest, Type type) {
    const Header h = hle::ReadGuest<Header>(guest);
    if (h.magic != kMagic || h.type != type || !h.host) {
        Fatal("nvn: %s is not an initialized NVN%s (header 0x%08x, type %u)", Describe(guest).c_str(), TypeName(type),
              h.magic, static_cast<u32>(h.type));
    }
    if (h.self != guest) {
        Fatal("nvn: the game copied an initialized NVN%s from 0x%llx to 0x%llx; copies are not supported yet",
              TypeName(type), (unsigned long long)h.self, (unsigned long long)guest);
    }
    return reinterpret_cast<Object*>(h.host);
}

Object* FindObject(u64 guest, Type type) {
    const Header h = hle::ReadGuest<Header>(guest);
    if (h.magic != kMagic || h.type != type || !h.host) return nullptr;
    return ObjectOf(guest, type);
}

void PutHeader(u64 guest, Type type, Object* o) {
    hle::WriteGuest(guest, Header{o ? kMagic : 0, type, guest, reinterpret_cast<u64>(o)});
}

std::unique_lock<std::recursive_mutex> LockScratch() {
    static std::recursive_mutex mu;
    return std::unique_lock(mu);
}

u64 RealProc(GuestContext* c, const std::string& name) {
    auto lock = LockScratch();
    static std::map<std::string, u64> cache;
    static u64 get_proc = 0;
    if (auto it = cache.find(name); it != cache.end()) return it->second;
    if (!g_bootstrap) {
        const auto found = FindSymbol("nvnBootstrapLoader");
        if (found.size() != 1) Fatal("nvn: the game's nvnBootstrapLoader is not in symbols.txt");
        g_bootstrap = found[0];
    }
    auto put = [](const std::string& s) {
        g_mem.Write(Scratch::kStrings, s.c_str(), s.size() + 1);
        return Scratch::kStrings;
    };
    if (!get_proc) {
        get_proc = hle::CallGuest(c, g_bootstrap, {put("nvnDeviceGetProcAddress")}, true);
        if (!get_proc) Fatal("nvn: the game's SDK has no nvnDeviceGetProcAddress");
    }
    const u64 addr = hle::CallGuest(c, get_proc, {0, put(name)}, true);
    cache[name] = addr;
    return addr;
}

s32 SdkDeviceInteger(GuestContext* c, u32 which) {
    auto lock = LockScratch();
    static u64 fn = RealProc(c, "nvnDeviceGetInteger");
    hle::WriteGuest<s32>(Scratch::kOut, 0);
    hle::CallGuest(c, fn, {Scratch::kDevice, which, Scratch::kOut}, true);
    return hle::ReadGuest<s32>(Scratch::kOut);
}

u64 SdkPure(GuestContext* c, const char* name, u64 a1, u64 a2, u64 a3) {
    static std::mutex mu;
    static std::map<std::string, u64> fns;
    u64 fn;
    {
        std::lock_guard lock(mu);
        auto [it, fresh] = fns.emplace(name, 0);
        if (fresh) it->second = RealProc(c, name);
        fn = it->second;
    }
    if (!fn) Fatal("nvn: the game's SDK has no %s", name);
    return hle::CallGuest(c, fn, {Scratch::kDevice, a1, a2, a3}, true);
}

void InstallShadows() {
    tap::AddShadow("nvnDeviceGetInteger", ShadowGetInteger);
    tap::AddShadow("nvnQueueInitialize", ShadowQueueInit);
    InstallMemoryShadows();
    InstallTextureShadows();
    InstallCaptureShadows();
    InstallProgramShadows();
}

} // namespace rt::nvn
