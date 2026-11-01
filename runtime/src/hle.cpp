// Host functions callable from guest code: registry, dispatch, overrides (hle.h).
#include "hle.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace rt::hle {

std::atomic<u32> g_override_count{0};

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

void Init(const std::string& slots_path) {
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
    const std::vector<u64> found = FindSymbol(symbol);
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
