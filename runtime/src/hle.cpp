// Host functions callable from guest code: registry, dispatch, overrides (hle.h).
#include "hle.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <map>
#include <unordered_map>
#include <vector>

namespace rt::hle {

std::atomic<u32> g_override_count{0};
bool g_trace_sites = false;

namespace {

constexpr size_t kMaxEntries = kHostFnRegionSize / kHostFnStride;

struct Entry {
    std::atomic<HostFn> fn{nullptr}; // published last: user and name are set before
    void* user = nullptr;
    std::string name;
};
// Entry i is the host function at kHostFnRegion + i * kHostFnStride; 0 is kHostReturn.
const std::unique_ptr<Entry[]> g_entries = std::make_unique<Entry[]>(kMaxEntries);
std::mutex g_register_mu;
size_t g_next_entry = 1;

struct Override {
    std::string symbol;
    u64 target = 0; // guest address of the lifted function
    u64 host = 0;   // its host address
    std::vector<u64> slots; // guest addresses of the GOT slots bound to it
};
// Installed before guest code runs, read-only afterwards.
constexpr size_t kMaxOverrides = 64;
std::array<Override, kMaxOverrides> g_overrides;

std::unordered_multimap<u64, u64> g_slots; // target -> slot (guest addresses), from slots.txt
bool g_have_slots = false;

const Override* FindOverride(u64 target) {
    const u32 n = g_override_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < n; ++i) {
        if (g_overrides[i].target == target) return &g_overrides[i];
    }
    return nullptr;
}

/// Points the GOT slots that rtld has bound to `o.target` at its host address.
/// A slot still pointing at the lazy-binding resolver is left alone: rtld checks
/// that at startup, and the resolver's first jump through it comes here again.
void RedirectSlots(const Override& o) {
    for (u64 slot : o.slots) {
        u64 expected = o.target;
        auto* p = reinterpret_cast<u64*>(g_mem.Raw(slot));
        if (__atomic_compare_exchange_n(p, &expected, o.host, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
            Log("hle: %s: GOT slot %s now points at the host function", o.symbol.c_str(),
                Describe(slot).c_str());
        }
    }
}

/// The runner behind CallOriginal and CallGuest: guest code from `target` until
/// it returns to kHostReturn.
void Run(GuestContext* c, u64 target, bool skip_override) {
    c->x[30] = kHostReturn;
    c->pc = target;
    if (skip_override) {
        const RtFunc* f = FindFunc(target);
        if (!f) {
            Fatal("hle: %s is not a lifted function start", Describe(target).c_str());
        }
        f->fn(c);
    }
    // As the top-level dispatch: a lifted function returns early when the guest
    // continues at an address none of its callers waits for.
    while (c->pc != kHostReturn) {
        rt_host_jump(c, c->pc);
    }
}

// --- RT_HLE_PASSTHROUGH: overrides that only log and call the original --------

struct Passthrough {
    std::string symbol;
    u64 target = 0;
    std::atomic<u64> calls{0};
};

void PassthroughFn(GuestContext* c, void* user) {
    auto* p = static_cast<Passthrough*>(user);
    const u64 n = p->calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 a0 = c->x[0], a1 = c->x[1], a2 = c->x[2], a3 = c->x[3], lr = c->x[30];
    CallOriginal(c, p->target);
    constexpr u64 kLoggedCalls = 20;
    if (n <= kLoggedCalls) {
        Log("hle: %s #%llu (0x%llx, 0x%llx, 0x%llx, 0x%llx) -> 0x%llx, called from %s%s", p->symbol.c_str(),
            (unsigned long long)n, (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
            (unsigned long long)a3, (unsigned long long)c->x[0], Describe(lr).c_str(),
            n == kLoggedCalls ? " (no more logged)" : "");
    }
}

void InstallPassthroughs(const char* list) {
    std::stringstream ss(list);
    std::string sym;
    while (std::getline(ss, sym, ',')) {
        if (sym.empty()) continue;
        auto* p = new Passthrough; // lives as long as the override
        p->symbol = sym;
        p->target = OverrideGuestFunction(sym, PassthroughFn, p);
        Log("hle: %s at %s passes through (RT_HLE_PASSTHROUGH)", sym.c_str(), Describe(p->target).c_str());
    }
}

// --- RT_HLE_TRACE: chosen functions' calls and chosen call sites' targets, with times ---------
//
// RT_HLE_TRACE=<item>[,<item>...] (measuring what the game does per frame, e.g. which battle checks count frames):
// - a function (a symbols.txt name or a guest address 0x...): every call that goes through a pointer - virtual calls,
//   the Blueprint VM's calls of native functions - logged with the time, the thread (its context, as a tag), x0-x3,
//   v0-v1 as floats, x0 and v0 after the call and the caller; <function>!x2: also the 8 bytes x2 pointed at, after
//   the call (a native function's Blueprint thunk, (object, FFrame&, result*): its result);
// - site:<return address>: the indirect call that returns there (a virtual call: the address after its BLR) - every
//   call: its target (a virtual function's implementation, to trace next), x0-x2 and v0 as a float.
// <function>!count: its calls only counted - every other function's line ends with the counts since the line before
//   (calls per frame of a function called too often to log, next to a once-a-frame function's lines).
// <function>!bt: its callers instead - each distinct chain of return addresses (the frame-pointer chain, up to 40)
//   logged once as chain #k, then counted as the !count items are: "chain#k=N" (which call paths run per frame).
// <item>@<n>: at most n lines for it (default 100000). Direct BL calls are not caught (hle.h).

struct Traced {
    std::string name;
    u64 target = 0;
    u64 limit = 100000;
    bool result_x2 = false;
    bool count_only = false;
    bool backtrace = false;
    std::atomic<u64> calls{0};
    std::atomic<u64> since{0}; // count_only: calls since the last line
};
std::vector<Traced*> g_counted; // the count_only items; installed before guest code runs

// !bt items' call chains: their ids and the calls of each since the last line.
std::mutex g_chain_mu;
std::map<std::vector<u64>, u32> g_chain_ids;
std::vector<u64> g_chain_since;

std::vector<u64> CallChain(GuestContext* c) {
    std::vector<u64> chain{c->x[30]};
    u64 fp = c->x[29];
    for (int i = 0; i < 40 && fp && !(fp & 15); ++i) {
        const Region* r = g_mem.Find(fp);
        if (!r || fp + 16 > r->base + r->size || !(r->perm & kRead)) break;
        u64 frame[2];
        std::memcpy(frame, g_mem.Raw(fp), 16);
        if (!frame[1]) break;
        chain.push_back(frame[1]);
        if (frame[0] <= fp) break; // stacks grow down: callers' frames lie above
        fp = frame[0];
    }
    return chain;
}

void NoteChain(GuestContext* c, const Traced& t) {
    std::vector<u64> chain = CallChain(c);
    std::lock_guard lock(g_chain_mu);
    auto [it, fresh] = g_chain_ids.emplace(std::move(chain), static_cast<u32>(g_chain_since.size()));
    if (fresh) {
        g_chain_since.push_back(0);
        std::string text;
        for (u64 ret : it->first) text += (text.empty() ? "" : " <- ") + Describe(ret);
        Log("hle trace: t=%.4f %s chain #%u: %s", TraceSeconds(), t.name.c_str(), it->second, text.c_str());
    }
    ++g_chain_since[it->second];
}

float LowFloat(u64 bits) {
    float f;
    const u32 b = static_cast<u32>(bits);
    std::memcpy(&f, &b, 4);
    return f;
}

void TracedFn(GuestContext* c, void* user) {
    auto* t = static_cast<Traced*>(user);
    if (t->backtrace) {
        NoteChain(c, *t);
        CallOriginal(c, t->target);
        return;
    }
    if (t->count_only) {
        t->since.fetch_add(1, std::memory_order_relaxed);
        CallOriginal(c, t->target);
        return;
    }
    const u64 n = t->calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 a0 = c->x[0], a1 = c->x[1], a2 = c->x[2], a3 = c->x[3], lr = c->x[30];
    const float f0 = LowFloat(c->vreg[0][0]), f1 = LowFloat(c->vreg[1][0]);
    const double at = TraceSeconds();
    CallOriginal(c, t->target);
    if (n <= t->limit) {
        char result[40] = "";
        if (t->result_x2) {
            u64 v = 0;
            if (const u8* p = g_mem.Access(a2, 8, kRead)) std::memcpy(&v, p, 8);
            std::snprintf(result, sizeof result, ", *x2 0x%llx", (unsigned long long)v);
        }
        std::string counts;
        for (Traced* k : g_counted) {
            counts += (counts.empty() ? "; counts " : " ") + k->name + "=" + std::to_string(k->since.exchange(0, std::memory_order_relaxed));
        }
        {
            std::lock_guard lock(g_chain_mu);
            for (size_t k = 0; k < g_chain_since.size(); ++k) {
                if (!g_chain_since[k]) continue;
                counts += (counts.empty() ? "; " : " ") + std::string("chain#") + std::to_string(k) + "=" + std::to_string(g_chain_since[k]);
                g_chain_since[k] = 0;
            }
        }
        Log("hle trace: t=%.4f %s #%llu thread %04llx (0x%llx, 0x%llx, 0x%llx, 0x%llx; %g, %g) -> 0x%llx, %g%s, from %s%s%s",
            at, t->name.c_str(), (unsigned long long)n, (unsigned long long)(reinterpret_cast<uintptr_t>(c) & 0xffff),
            (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)a3, f0, f1,
            (unsigned long long)c->x[0], LowFloat(c->vreg[0][0]), result, Describe(lr).c_str(),
            n == t->limit ? " (no more logged)" : "", counts.c_str());
    }
}

struct Site {
    u64 ret = 0;
    u64 limit = 100000;
    std::mutex mu;
    std::unordered_map<u64, u64> targets; // target -> calls
    u64 lines = 0;
};
std::vector<std::unique_ptr<Site>> g_sites; // installed before guest code runs, read-only afterwards

void InstallTrace(const char* list) {
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        u64 limit = 100000;
        bool result_x2 = false, count_only = false, backtrace = false;
        if (const size_t at = item.find("!x2"); at != std::string::npos) result_x2 = true, item.erase(at, 3);
        if (const size_t at = item.find("!count"); at != std::string::npos) count_only = true, item.erase(at, 6);
        if (const size_t at = item.find("!bt"); at != std::string::npos) backtrace = true, item.erase(at, 3);
        if (const size_t at = item.find('@'); at != std::string::npos) {
            limit = std::strtoull(item.c_str() + at + 1, nullptr, 10);
            item.resize(at);
        }
        if (item.rfind("site:", 0) == 0) {
            auto s = std::make_unique<Site>();
            s->ret = std::strtoull(item.c_str() + 5, nullptr, 16);
            s->limit = limit;
            Log("hle trace: the indirect call returning to %s: its targets (RT_HLE_TRACE)", Describe(s->ret).c_str());
            g_sites.push_back(std::move(s));
            continue;
        }
        auto* t = new Traced; // lives as long as the override
        t->result_x2 = result_x2;
        t->count_only = count_only;
        t->backtrace = backtrace;
        t->name = item;
        t->limit = limit;
        t->target = OverrideGuestFunction(item, TracedFn, t);
        if (count_only) g_counted.push_back(t);
        Log("hle trace: %s at %s: %s (RT_HLE_TRACE)", item.c_str(), Describe(t->target).c_str(),
            backtrace ? "call chains counted" : count_only ? "calls counted" : "every call");
    }
    g_trace_sites = !g_sites.empty();
}

// --- RT_HLE_SELFTEST=1: the binding, the dispatch and CallGuest, without the game -----

u64 TestSum(GuestContext*, u64 a, u32 b, s32 neg, u8 d, u64 e, u64 f, u64 g, u64 h, u64 on_stack, float x,
            double y, bool flag) {
    return a + b + static_cast<u64>(neg) + d + e + f + g + h + on_stack + static_cast<u64>(x * 10) +
           static_cast<u64>(y * 100) + (flag ? 1000 : 0);
}

float TestHalf(GuestContext*, float v) { return v / 2; }

u64 g_test_sum_addr = 0;

/// Host code calling guest code calling host code: CallGuest through the region.
u64 TestNested(GuestContext* c, u64 v) {
    const u64 before = c->x[0];
    // TestSum's integer arguments 9 and 10 (on_stack, flag) come from the stack.
    c->x[31] -= 16;
    WriteGuest<u64>(c->x[31], 100000);
    WriteGuest<u64>(c->x[31] + 8, 1);
    c->vreg[0][0] = std::bit_cast<u32>(1.5f); // x * 10 = 15
    c->vreg[1][0] = std::bit_cast<u64>(0.25); // y * 100 = 25
    const u64 r = CallGuest(c, g_test_sum_addr, {v, 2, static_cast<u64>(-3), 0x104, 5, 6, 7, 8});
    c->x[31] += 16;
    if (c->x[0] != before) Fatal("hle self-test: CallGuest did not restore x0");
    return r;
}

void SelfTest() {
    g_test_sum_addr = RegisterHostFunction("selftest.sum", Bind<&TestSum>());
    const u64 half = RegisterHostFunction("selftest.half", Bind<&TestHalf>());
    const u64 nested = RegisterHostFunction("selftest.nested", Bind<&TestNested>());

    // A context on the (still unused) main stack, as a guest thread would have.
    auto ctx = std::make_unique<GuestContext>();
    GuestContext* c = ctx.get();
    c->x[31] = kStackRegion + kMainStackSize - 0x100;
    const u64 sp = c->x[31];

    c->x[0] = 7;
    const u64 r = CallGuest(c, nested, {1});
    // u8 d = 0x104 -> 4; x = 1.5 -> 15; y = 0.25 -> 25; flag -> 1000.
    const u64 want = 1 + 2 - 3 + 4 + 5 + 6 + 7 + 8 + 100000 + 15 + 25 + 1000;
    if (r != want) Fatal("hle self-test: nested sum is %llu, want %llu", (unsigned long long)r, (unsigned long long)want);
    if (c->x[0] != 7 || c->x[31] != sp) Fatal("hle self-test: CallGuest did not restore the caller's registers");

    c->vreg[0][0] = std::bit_cast<u32>(3.0f);
    c->vreg[0][1] = ~0ULL;
    c->x[30] = kHostReturn;
    rt_host_jump(c, half); // as lifted code reaches it: an indirect call
    if (c->pc != kHostReturn || std::bit_cast<float>(static_cast<u32>(c->vreg[0][0])) != 1.5f || c->vreg[0][1] != 0) {
        Fatal("hle self-test: float result or return address wrong");
    }
    if (HostFunctionName(half) != "selftest.half" || Describe(half).find("host:selftest.half") == std::string::npos) {
        Fatal("hle self-test: names wrong (%s)", Describe(half).c_str());
    }
    Log("hle: self-test passed (binding, dispatch, nested CallGuest, float result)");
}

} // namespace

double TraceSeconds() {
    static const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void Init(const std::string& slots_path) {
    TraceSeconds(); // (its zero: before guest code runs)
    g_mem.Map(kHostFnRegion, kHostFnRegionSize, 0, kMemInaccessible, "host functions");
    g_mem.Map(kHostScratchRegion, kHostScratchRegionSize, kRead | kWrite, kMemNormal, "host scratch");
    if (std::ifstream f(slots_path); f) {
        u64 slot, target;
        while (f >> std::hex >> slot >> target) {
            g_slots.emplace(kImageBase + target, kImageBase + slot);
        }
        g_have_slots = true;
    }
    if (std::getenv("RT_HLE_SELFTEST")) {
        SelfTest();
    }
    if (const char* list = std::getenv("RT_HLE_PASSTHROUGH")) {
        InstallPassthroughs(list);
    }
    if (const char* list = std::getenv("RT_HLE_TRACE")) {
        InstallTrace(list);
    }
}

void NoteSite(GuestContext* c, u64 target) {
    for (const auto& s : g_sites) {
        if (s->ret != c->x[30]) continue;
        std::lock_guard lock(s->mu);
        const u64 n = ++s->targets[target];
        if (s->lines < s->limit) {
            ++s->lines;
            Log("hle trace: t=%.4f site %s -> %s%s (0x%llx, 0x%llx, 0x%llx; %g)", TraceSeconds(), Describe(s->ret).c_str(),
                Describe(target).c_str(), n == 1 ? " (new)" : "", (unsigned long long)c->x[0],
                (unsigned long long)c->x[1], (unsigned long long)c->x[2], LowFloat(c->vreg[0][0]));
        }
    }
}

u64 RegisterHostFunction(const std::string& name, HostFn fn, void* user) {
    std::lock_guard lock(g_register_mu);
    if (g_next_entry == kMaxEntries) {
        Fatal("hle: more than %zu host functions", kMaxEntries - 1);
    }
    Entry& e = g_entries[g_next_entry];
    e.user = user;
    e.name = name;
    e.fn.store(fn, std::memory_order_release);
    return kHostFnRegion + g_next_entry++ * kHostFnStride;
}

u64 OverrideGuestFunction(const std::string& symbol, HostFn fn, void* user) {
    const std::vector<u64> found = symbol.rfind("0x", 0) == 0 ? std::vector<u64>{std::strtoull(symbol.c_str(), nullptr, 16)}
                                                              : FindSymbol(symbol);
    if (found.size() != 1) {
        Fatal("hle: cannot override %s: %zu symbols.txt entries have that name", symbol.c_str(), found.size());
    }
    const u64 target = found[0];
    if (!FindFunc(target)) {
        Fatal("hle: cannot override %s: %s is not a lifted function start", symbol.c_str(), Describe(target).c_str());
    }
    const u32 n = g_override_count.load(std::memory_order_relaxed);
    if (FindOverride(target)) Fatal("hle: %s is overridden twice", symbol.c_str());
    if (n == kMaxOverrides) Fatal("hle: more than %zu overrides", kMaxOverrides);
    if (!g_have_slots) {
        Log("hle: no slots.txt: PLT calls to %s whose slot is bound may bypass the override", symbol.c_str());
    }
    Override& o = g_overrides[n];
    o.symbol = symbol;
    o.target = target;
    o.host = RegisterHostFunction(symbol, fn, user);
    auto [lo, hi] = g_slots.equal_range(target);
    for (auto it = lo; it != hi; ++it) {
        o.slots.push_back(it->second);
    }
    g_override_count.store(n + 1, std::memory_order_release);
    MarkOverridden(FindFunc(target)); // after the override is findable: rt_jump looks for it when marked
    RedirectSlots(o); // slots rtld already bound (an override installed late)
    return target;
}

std::string HostFunctionName(u64 va) {
    if (!IsHostAddress(va)) return "";
    const u64 off = va - kHostFnRegion;
    if (off == 0) return "return to host";
    const u64 i = off / kHostFnStride;
    if (off % kHostFnStride || i >= kMaxEntries || !g_entries[i].fn.load(std::memory_order_acquire)) return "";
    return g_entries[i].name;
}

void CallHost(GuestContext* c, u64 va) {
    if (va == kHostReturn) { // lifted code returning to its host caller through a branch
        c->pc = va;
        return;
    }
    const u64 off = va - kHostFnRegion;
    const u64 i = off / kHostFnStride;
    const HostFn fn = off % kHostFnStride == 0 && i < kMaxEntries ? g_entries[i].fn.load(std::memory_order_acquire)
                                                                  : nullptr;
    if (!fn) {
        Fatal("branch to %s, which is no host function", Describe(va).c_str());
    }
    const u64 lr = c->x[30];
    fn(c, g_entries[i].user);
    c->pc = lr;
}

bool CallOverride(GuestContext* c, u64 target) {
    const Override* o = FindOverride(target);
    if (!o) return false;
    RedirectSlots(*o);
    CallHost(c, o->host);
    return true;
}

bool IsOverridden(u64 target) { return FindOverride(target) != nullptr; }

u64 StackArg(const GuestContext* c, int slot) { return ReadGuest<u64>(c->x[31] + 8 * static_cast<u64>(slot)); }

void CallOriginal(GuestContext* c, u64 target) {
    const u64 lr = c->x[30];
    Run(c, target, true);
    c->x[30] = lr;
}

u64 CallGuest(GuestContext* c, u64 target, std::initializer_list<u64> args, bool skip_override) {
    return CallGuestFp(c, target, args, {}, nullptr, skip_override);
}

u64 CallGuestFp(GuestContext* c, u64 target, std::initializer_list<u64> args, std::initializer_list<u64> fp_args,
                u64* v0, bool skip_override) {
    if (args.size() > 8 || fp_args.size() > 8) {
        Fatal("hle: CallGuest with %zu + %zu arguments (at most 8 each)", args.size(), fp_args.size());
    }
    // What the AAPCS64 lets a callee change: x0-x18, x29 and x30 by convention
    // only, sp, the flags, the vector registers (v8-v15 only in their upper
    // halves; all are kept), and pc.
    u64 x[19];
    std::memcpy(x, c->x, sizeof x);
    u64 v[32][2];
    std::memcpy(v, c->vreg, sizeof v);
    const u64 fp = c->x[29], lr = c->x[30], sp = c->x[31], pc = c->pc;
    const u8 n = c->n, z = c->z, cf = c->c, vf = c->v;

    size_t i = 0;
    for (u64 a : args) c->x[i++] = a;
    i = 0;
    for (u64 a : fp_args) {
        c->vreg[i][0] = a;
        c->vreg[i++][1] = 0;
    }
    Run(c, target, skip_override);
    const u64 result = c->x[0];
    if (v0) *v0 = c->vreg[0][0];

    std::memcpy(c->x, x, sizeof x);
    std::memcpy(c->vreg, v, sizeof v);
    c->x[29] = fp;
    c->x[30] = lr;
    c->x[31] = sp;
    c->pc = pc;
    c->n = n;
    c->z = z;
    c->c = cf;
    c->v = vf;
    return result;
}

} // namespace rt::hle
