// Host functions callable from guest code.
//
// A host function gets a fake guest address in kHostFnRegion (rt.h), a region
// that is reserved in the guest map and never accessible. Guest code reaches it
// as it reaches any function through a pointer: the indirect branch lands in
// rt_jump, which runs the host function and continues at the caller's x30, as a
// lifted function's RET does.
//
// An override replaces a lifted function by its symbols.txt name. It catches
// every call that goes through a pointer: rt_jump's, and PLT calls. The lifter
// turns a PLT call whose GOT slot it bound statically into a guarded direct call
// (`if (x17 == target) f_target(c)`), so the override also redirects the slots
// bound to the function (slots.txt) to its host address once rtld has filled
// them: rtld binds lazily, and the first call through a slot is the resolver's
// jump to the target, which rt_jump sees. Direct B/BL between lifted functions
// are not caught: check that a target has none (search the lifted code for
// direct calls to it) before overriding it.
#pragma once

#include "rt.h"

#include <array>
#include <bit>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace rt::hle {

constexpr u64 kHostFnStride = 16;
/// Entry 0 of the region: the return address host code hands the lifted code it
/// runs (CallGuest, CallOriginal). Never a function.
constexpr u64 kHostReturn = kHostFnRegion;

using HostFn = void (*)(GuestContext* c, void* user);

/// Reserves the region and the scratch region (kHostScratchRegion), reads `slots_path` (the lifter's slots.txt; may be
/// missing) and installs the overrides named in RT_HLE_PASSTHROUGH. After the
/// image is mapped and symbols are loaded, before guest code runs.
void Init(const std::string& slots_path);

/// Gives `fn` a host address; thread-safe. `name` is for logs and Describe.
u64 RegisterHostFunction(const std::string& name, HostFn fn, void* user = nullptr);

/// Diverts the lifted function named `symbol` (exactly one symbols.txt entry)
/// to `fn`; returns the function's guest address. Before guest code runs.
u64 OverrideGuestFunction(const std::string& symbol, HostFn fn, void* user = nullptr);

inline bool IsHostAddress(u64 va) { return va - kHostFnRegion < kHostFnRegionSize; }
/// "name" of a host address, or "" for none.
std::string HostFunctionName(u64 va);

// --- Dispatch (rt_jump) ----------------------------------------------------

/// Runs the host function at `va` and returns to the caller's x30.
void CallHost(GuestContext* c, u64 va);

extern std::atomic<u32> g_override_count;
/// Runs the override of `target`, if it has one.
bool CallOverride(GuestContext* c, u64 target);
inline bool MaybeCallOverride(GuestContext* c, u64 target) {
    return g_override_count.load(std::memory_order_relaxed) != 0 && CallOverride(c, target);
}
bool IsOverridden(u64 target);

// --- Calling lifted code from host code ------------------------------------

/// Runs the lifted function at `target`, skipping its override, with the
/// registers as they are: a pass-through. Afterwards they hold its results, as
/// after any call; x30 is kept.
void CallOriginal(GuestContext* c, u64 target);

/// Calls guest code at `target` with integer arguments in x0..x7 and returns
/// its x0. Every register a callee may change is restored, so the host
/// function's own arguments are still there afterwards. Overrides apply,
/// unless `skip_override` (to reach the lifted function an override replaces).
u64 CallGuest(GuestContext* c, u64 target, std::initializer_list<u64> args = {}, bool skip_override = false);
/// CallGuest with floating-point arguments too: the low 64 bits of v0..v7 (a
/// float's bits zero-extended, upper halves zeroed); stores v0's low 64 bits
/// after the call in `*v0` if given (a float or double result).
u64 CallGuestFp(GuestContext* c, u64 target, std::initializer_list<u64> args, std::initializer_list<u64> fp_args,
                u64* v0, bool skip_override = false);

// --- Arguments and results (AAPCS64) ----------------------------------------
// Integers and pointers in x0..x7, floats in v0..v7 (each independently), the
// rest on the stack in 8-byte slots from sp. Results in x0 or v0.

inline u64 ArgX(const GuestContext* c, int i) { return c->x[i]; }
inline float ArgF(const GuestContext* c, int i) { return std::bit_cast<float>(static_cast<u32>(c->vreg[i][0])); }
inline double ArgD(const GuestContext* c, int i) { return std::bit_cast<double>(c->vreg[i][0]); }
/// The `slot`th 8-byte stack argument.
u64 StackArg(const GuestContext* c, int slot);

inline void RetX(GuestContext* c, u64 v) { c->x[0] = v; }
inline void RetF(GuestContext* c, float v) {
    c->vreg[0][0] = std::bit_cast<u32>(v); // a scalar write clears the rest of the register
    c->vreg[0][1] = 0;
}
inline void RetD(GuestContext* c, double v) {
    c->vreg[0][0] = std::bit_cast<u64>(v);
    c->vreg[0][1] = 0;
}

/// Typed guest memory (checked: a bad address stops the run with a diagnostic).
template <typename T>
T ReadGuest(u64 va) {
    static_assert(std::is_trivially_copyable_v<T>);
    T v;
    g_mem.Read(va, &v, sizeof v);
    return v;
}
template <typename T>
void WriteGuest(u64 va, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    g_mem.Write(va, &v, sizeof v);
}
inline std::string ReadGuestString(u64 va, u64 max = 4096) { return g_mem.ReadString(va, max); }

// --- Typed binding ------------------------------------------------------------
// Bind<F>() turns `R F(GuestContext*, A...)` into a HostFn: each A is read from
// the next integer register, float register or stack slot by the AAPCS64, and R
// is returned in x0 or v0. A and R: integers, enums, bool, float, double; guest
// pointers are u64.

namespace detail {

template <typename T>
constexpr bool kIsFloat = std::is_floating_point_v<T>;
template <typename T>
constexpr bool kIsScalar = std::is_integral_v<T> || std::is_enum_v<T> || kIsFloat<T>;

struct Slot {
    u8 kind; // 0 integer register, 1 float register, 2 stack
    u8 index;
};

template <typename... A>
constexpr std::array<Slot, sizeof...(A)> Assign() {
    std::array<Slot, sizeof...(A)> s{};
    u8 nx = 0, nv = 0, ns = 0;
    size_t i = 0;
    ((s[i++] = kIsFloat<A> ? (nv < 8 ? Slot{1, nv++} : Slot{2, ns++})
                           : (nx < 8 ? Slot{0, nx++} : Slot{2, ns++})),
     ...);
    return s;
}

template <typename T>
T Get(const GuestContext* c, Slot s) {
    static_assert(kIsScalar<T>, "host function arguments: integers, enums, bool, float, double");
    u64 bits = s.kind == 0 ? c->x[s.index] : s.kind == 1 ? c->vreg[s.index][0] : StackArg(c, s.index);
    if constexpr (std::is_same_v<T, float>) {
        return std::bit_cast<float>(static_cast<u32>(bits));
    } else if constexpr (std::is_same_v<T, double>) {
        return std::bit_cast<double>(bits);
    } else if constexpr (std::is_same_v<T, bool>) {
        return (bits & 0xFF) != 0;
    } else {
        return static_cast<T>(bits);
    }
}

template <auto F, typename Sig>
struct Binder;

template <auto F, typename R, typename... A>
struct Binder<F, R (*)(GuestContext*, A...)> {
    static void Call(GuestContext* c, void*) { call(c, std::index_sequence_for<A...>{}); }
    template <size_t... I>
    static void call(GuestContext* c, std::index_sequence<I...>) {
        static constexpr auto slots = Assign<A...>();
        if constexpr (std::is_void_v<R>) {
            F(c, Get<A>(c, slots[I])...);
        } else {
            static_assert(kIsScalar<R>, "host function results: integers, enums, bool, float, double");
            const R r = F(c, Get<A>(c, slots[I])...);
            if constexpr (std::is_same_v<R, float>) {
                RetF(c, r);
            } else if constexpr (std::is_same_v<R, double>) {
                RetD(c, r);
            } else {
                RetX(c, static_cast<u64>(r));
            }
        }
    }
};

} // namespace detail

template <auto F>
constexpr HostFn Bind() {
    return &detail::Binder<F, decltype(F)>::Call;
}

} // namespace rt::hle
