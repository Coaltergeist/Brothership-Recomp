#include "rt.h"
#include "diag.h"
#include "gpu.h"
#include "hle.h"
#include "options.h"
#include "nvn/layer.h"
#include "nvn/tap.h"
#include "platform/platform.h"
#include "services.h"
#include "threads.h"
#include "window.h"
#include "writetrack.h"

#include <thread>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>

extern "C" {
thread_local uint64_t rt_trace_pc = 0;
}

// The lift's memory mode: `nxrecomp --direct-mem` defines nx_mem_direct_base in nx_table.c (its lifted code
// reaches guest memory at that host address, old_plans/FASTMEM_PLAN.md Phase 2); other lifts do not define it.
#ifdef _WIN32
extern "C" const unsigned long long nx_mem_direct_base_none = 0;
#pragma comment(linker, "/alternatename:nx_mem_direct_base=nx_mem_direct_base_none")
extern "C" const unsigned long long nx_mem_direct_base;
static unsigned long long LiftDirectBase() { return nx_mem_direct_base; }
#else
extern "C" __attribute__((weak)) const unsigned long long nx_mem_direct_base;
static unsigned long long LiftDirectBase() { return &nx_mem_direct_base ? nx_mem_direct_base : 0; }
#endif

namespace rt {

Memory g_mem;
namespace {
bool g_direct_mem = false; // the lift accesses guest memory directly (LiftDirectBase)
}
// RT_TRACE_SVC=0: no log line per SVC and IPC call (on by default: the log is the main debugging record);
// RT_TRACE_SVC=first: each SVC trace site and each IPC command logged its first 100 times, IPC calls that fail
// always (long play sessions: the full trace is ~11 MB a minute; unknown commands stop the run either way).
bool g_trace_svc = [] {
    const char* e = std::getenv("RT_TRACE_SVC");
    return !(e && *e == '0');
}();
bool g_trace_all = [] {
    const char* e = std::getenv("RT_TRACE_SVC");
    if (e && std::strcmp(e, "first") == 0) {
        std::fputs("[rt] SVC and IPC trace: the first 100 lines of each kind, then failing IPC calls (RT_TRACE_SVC=first)\n", stderr);
        return false;
    }
    return true;
}();

namespace {

std::map<u64, std::string> g_symbols; // module offset -> name, from symbols.txt

void VLog(const char* prefix, const char* fmt, va_list ap) {
    platform::LockStderr(); // one line at a time: guest threads log in parallel
    std::fputs(prefix, stderr);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    platform::UnlockStderr();
}

// --- RecompHostMem callbacks: every guest access from generated code -------

// The exclusive monitor is per thread: LDXR remembers the address and the value
// it read, and STXR stores only if memory still holds that value, as one
// compare-and-swap (other threads run guest code in parallel). Like dynarmic
// and drippu, this accepts the ABA difference from a real monitor.
thread_local u64 g_excl_addr = ~0ULL, g_excl_lo = 0, g_excl_hi = 0;

u64 RawLoad(u64 va, u32 size) {
    u64 v = 0;
    std::memcpy(&v, g_mem.Access(va, size, kRead), size);
    return v;
}

u64 Load(void*, u64 va, u32 size) {
    // In the global-lock mode (RT_GIL=1), guest loads are a preemption point too:
    // a spin-wait may poll memory with no call at all (UE4's task wait re-scans
    // its tasks' "done" flags with LDAR until another thread sets them). Never
    // inside an LDXR/STXR pair. A no-op when threads run in parallel.
    thread_local u32 loads = 0;
    if (++loads >= 0x10000) {
        loads = 0;
        if (g_excl_addr == ~0ULL) PreemptNow();
    }
    return RawLoad(va, size);
}

void Store(void*, u64 va, u32 size, u64 value) {
    std::memcpy(g_mem.Access(va, size, kWrite), &value, size);
}

template <typename T>
bool CasAt(u8* p, u64 expected, u64 desired) {
    T e = static_cast<T>(expected);
    return __atomic_compare_exchange_n(reinterpret_cast<T*>(p), &e, static_cast<T>(desired), false,
                                       __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

bool Cas(u8* p, u32 size, u64 expected, u64 desired) {
    switch (size) {
    case 1: return CasAt<u8>(p, expected, desired);
    case 2: return CasAt<u16>(p, expected, desired);
    case 4: return CasAt<u32>(p, expected, desired);
    default: return CasAt<u64>(p, expected, desired);
    }
}

/// 16-byte compare-and-swap (LDXP/STXP of two 64-bit registers).
bool Cas128(u8* p, u64 elo, u64 ehi, u64 nlo, u64 nhi) {
    bool ok;
    __asm__ __volatile__("lock cmpxchg16b %1"
                         : "=@ccz"(ok), "+m"(*reinterpret_cast<volatile unsigned __int128*>(p)), "+a"(elo), "+d"(ehi)
                         : "b"(nlo), "c"(nhi)
                         : "memory");
    return ok;
}

u64 ExclLoad(void*, u64 va, u32 size) {
    g_excl_addr = va;
    g_excl_lo = RawLoad(va, size);
    return g_excl_lo;
}

u32 ExclStore(void*, u64 va, u32 size, u64 value) {
    if (g_excl_addr != va) {
        return 1;
    }
    g_excl_addr = ~0ULL;
    return Cas(g_mem.Access(va, size, kRead | kWrite), size, g_excl_lo, value) ? 0 : 1;
}

void ClearExcl(void*) { g_excl_addr = ~0ULL; }

void ExclLoadPair(void*, u64 va, u32 size, u64* lo, u64* hi) {
    g_excl_addr = va;
    *lo = g_excl_lo = RawLoad(va, size);
    *hi = g_excl_hi = RawLoad(va + size, size);
}

u32 ExclStorePair(void*, u64 va, u32 size, u64 lo, u64 hi) {
    if (g_excl_addr != va) {
        return 1;
    }
    g_excl_addr = ~0ULL;
    u8* p = g_mem.Access(va, size * 2, kRead | kWrite);
    if (size == 4) {
        const u64 expected = (g_excl_lo & 0xFFFF'FFFFULL) | (g_excl_hi << 32);
        return CasAt<u64>(p, expected, (lo & 0xFFFF'FFFFULL) | (hi << 32)) ? 0 : 1;
    }
    return Cas128(p, g_excl_lo, g_excl_hi, lo, hi) ? 0 : 1;
}

u64 ReadCntpct(void*) {
    // Polling loops read the counter, and their calls are direct (no rt_jump),
    // so this is also a preemption point: without it a thread spinning on
    // another (FlushAsyncLoading waiting for IoDispatcher) never lets it run.
    MaybePreempt();
    // The Switch's system counter runs at 19.2 MHz.
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
    return static_cast<u64>(ns) * 192 / 10000;
}

RecompHostMem g_host_mem;

void LoadSymbols(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        const auto sp = line.find(' ');
        if (sp != std::string::npos) {
            g_symbols.emplace(std::stoull(line.substr(0, sp), nullptr, 16), line.substr(sp + 1));
        }
    }
}

} // namespace

namespace {
struct FuncIndex {
    u64 pages = 0;
    std::unique_ptr<u32[]> first;                    // [page]: the first function at or after the page's start
    std::unique_ptr<std::atomic<u8>[]> overridden;   // [function]: has an override
};
const FuncIndex& Index() {
    static const FuncIndex index = [] {
        FuncIndex x;
        const u64 last = rt_func_count ? rt_funcs[rt_func_count - 1].offset : 0;
        x.pages = (std::max<u64>(rt_image_size, last + 1) >> 12) + 1;
        x.first = std::make_unique<u32[]>(x.pages + 1);
        u64 f = 0;
        for (u64 p = 0; p <= x.pages; ++p) {
            while (f < rt_func_count && rt_funcs[f].offset < (p << 12)) ++f;
            x.first[p] = static_cast<u32>(f);
        }
        x.overridden = std::make_unique<std::atomic<u8>[]>(rt_func_count);
        return x;
    }();
    return index;
}
} // namespace

const RtFunc* FindFunc(u64 va) {
    if (va < kImageBase) {
        return nullptr;
    }
    const u64 off = va - kImageBase;
    const FuncIndex& x = Index();
    const u64 p = off >> 12;
    if (p >= x.pages) return nullptr;
    const RtFunc* end = rt_funcs + x.first[p + 1];
    const RtFunc* it = std::lower_bound(rt_funcs + x.first[p], end, off,
                                        [](const RtFunc& f, u64 v) { return f.offset < v; });
    return it != end && it->offset == off ? it : nullptr;
}

// rt_jump's per-thread cache of recent targets (lifted functions without an override), direct-mapped; a new
// override empties every thread's (g_jump_gen).
std::atomic<u64> g_jump_gen{1};
struct JumpCache {
    static constexpr u32 kEntries = 4096;
    u64 gen = 0;
    struct Entry {
        u64 target;
        const RtFunc* f;
    } e[kEntries];
};
thread_local JumpCache* t_jump_cache = nullptr;

void MarkOverridden(const RtFunc* f) {
    Index().overridden[f - rt_funcs].store(1, std::memory_order_release);
    g_jump_gen.fetch_add(1, std::memory_order_release);
}

bool Overridden(const RtFunc* f) {
    return Index().overridden[f - rt_funcs].load(std::memory_order_acquire) != 0;
}

std::vector<u64> FindSymbol(const std::string& name) {
    std::vector<u64> found;
    for (const auto& [off, sym] : g_symbols) {
        if (sym == name) found.push_back(kImageBase + off);
    }
    return found;
}

// --- Memory ------------------------------------------------------------------

void Memory::Init() {
    host_ = platform::ReserveWindow(kWindowSize, kHostWindowBase);
    if (!host_) {
        Fatal("cannot reserve 0x%llx bytes of guest address space at host address 0x%llx (in use?)",
              (unsigned long long)kWindowSize, (unsigned long long)kHostWindowBase);
    }
}

namespace {
// RT_MEM_STATS=1: the guest map's own costs per Map/MapMerged call (the platform layer reports its part), as
// "map:" log lines every 500 calls: time waiting for the exclusive lock, inserting the region, committing host
// memory, refreshing the page table, merging, and the region count; the same for UnmapRange ("unmap:":
// lock, erasing / re-inserting regions, page table, device views, giving the host pages back).
struct MapStats {
    std::atomic<u64> n{0}, lock{0}, insert{0}, commit{0}, refresh{0}, merge{0};
};
MapStats g_map_stats;
struct UnmapStats {
    std::atomic<u64> n{0}, bytes{0}, lock{0}, regions{0}, refresh{0}, device{0}, zero{0};
};
UnmapStats g_unmap_stats;
bool MapStatsOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_MEM_STATS");
        return e && *e && *e != '0';
    }();
    return on;
}
u64 MapNow() { return MapStatsOn() ? static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch().count()) : 0; }

} // namespace

void Memory::Map(u64 base, u64 size, u32 perm, u32 type, const std::string& name) {
    if (base + size > kWindowBase + kWindowSize || base + size < base) {
        Fatal("mapping %s at 0x%llx+0x%llx is outside the guest window", name.c_str(),
              (unsigned long long)base, (unsigned long long)size);
    }
    // Keep regions sorted by base: insert in place instead of re-sorting (the game maps
    // and unmaps a page every frame).
    const u64 t0 = MapNow();
    std::unique_lock lock(mu_);
    const u64 t1 = MapNow();
    auto it = std::upper_bound(regions_.begin(), regions_.end(), base,
                               [](u64 v, const Region& r) { return v < r.base; });
    regions_.insert(it, {base, size, perm, type, name});
    const u64 t2 = MapNow();
    platform::CommitWindow(Raw(base), size);
    writetrack::Forget(base, size); // fresh pages: anything watched there before has changed
    const u64 t3 = MapNow();
    RefreshPagesLocked(base, base + size);
    gen_.fetch_add(1, std::memory_order_release);
    if (MapStatsOn()) {
        const u64 t4 = MapNow();
        g_map_stats.lock += t1 - t0, g_map_stats.insert += t2 - t1, g_map_stats.commit += t3 - t2, g_map_stats.refresh += t4 - t3;
    }
}

void Memory::MapMerged(u64 base, u64 size, u32 perm, u32 type, const std::string& name) {
    Map(base, size, perm, type, name);
    const u64 t0 = MapNow();
    std::unique_lock lock(mu_);
    struct Done {
        u64 t0;
        size_t regions;
        ~Done() {
            if (!MapStatsOn()) return;
            g_map_stats.merge += MapNow() - t0;
            const u64 n = ++g_map_stats.n;
            if (n % 500) return;
            auto ms = [](std::atomic<u64>& v) { return v.exchange(0) / 1e6; };
            const double l = ms(g_map_stats.lock), i = ms(g_map_stats.insert), c = ms(g_map_stats.commit),
                         r = ms(g_map_stats.refresh), m = ms(g_map_stats.merge);
            Log("map: %llu merged maps; last 500: lock %.1f ms, insert %.1f ms, commit %.1f ms, pages %.1f ms, merge %.1f ms; %zu regions",
                (unsigned long long)n, l, i, c, r, m, regions);
        }
    } done{t0, regions_.size()};
    auto same = [&](const Region& r) { return r.perm == perm && r.type == type && r.name == name; };
    auto it = std::lower_bound(regions_.begin(), regions_.end(), base,
                               [](const Region& r, u64 v) { return r.base < v; }); // the new region
    if (it == regions_.end() || it->base != base) return;
    if (auto next = std::next(it); next != regions_.end() && next->base == base + size && same(*next)) {
        it->size += next->size;
        regions_.erase(next);
    }
    if (it != regions_.begin()) {
        if (auto prev = std::prev(it); prev->base + prev->size == base && same(*prev)) {
            prev->size += it->size;
            regions_.erase(it);
        }
    }
    // Merging changes no page: the page table and the generation are already up to date (Map).
}

std::pair<size_t, size_t> Memory::OverlapLocked(u64 lo, u64 hi) const {
    auto first = std::upper_bound(regions_.begin(), regions_.end(), lo,
                                  [](u64 v, const Region& r) { return v < r.base; });
    if (first != regions_.begin() && std::prev(first)->base + std::prev(first)->size > lo) --first;
    auto last = std::lower_bound(first, regions_.end(), hi, [](const Region& r, u64 v) { return r.base < v; });
    return {static_cast<size_t>(first - regions_.begin()), static_cast<size_t>(last - regions_.begin())};
}

u64 Memory::NextRegionBase(u64 va) const {
    std::shared_lock lock(mu_);
    auto it = std::upper_bound(regions_.begin(), regions_.end(), va, [](u64 v, const Region& r) { return v < r.base; });
    return it == regions_.end() ? kAddressSpaceEnd : it->base;
}

void Memory::Resize(const std::string& name, u64 size) {
    std::unique_lock lock(mu_);
    for (auto& r : regions_) {
        if (r.name == name) {
            const u64 old_size = r.size;
            r.size = size;
            if (size > old_size) platform::CommitWindow(Raw(r.base + old_size), size - old_size);
            if (size < old_size) writetrack::Forget(r.base + size, old_size - size); // before the zero-fill
            if (size < old_size) platform::ZeroWindow(Raw(r.base + size), old_size - size); // unmapped again
            if (size > old_size) writetrack::Forget(r.base + old_size, size - old_size);
            RefreshPagesLocked(r.base, r.base + std::max(old_size, size));
        }
    }
    gen_.fetch_add(1, std::memory_order_release);
}

bool Memory::Protect(u64 base, u64 size, u32 perm) {
    // In place: only the regions the range overlaps are split and changed (the map holds tens of thousands
    // of regions while loading; copying it per call was the cost, ~1.8 MB per call).
    std::unique_lock lock(mu_);
    const auto [a, b] = OverlapLocked(base, base + size);
    u64 covered = 0;
    for (size_t i = a; i < b; ++i) {
        const Region& r = regions_[i];
        covered += std::min(base + size, r.base + r.size) - std::max(base, r.base);
    }
    if (covered != size) {
        return false; // part of the range is unmapped
    }
    std::vector<Region> pieces;
    pieces.reserve(b - a + 2);
    for (size_t i = a; i < b; ++i) {
        const Region& r = regions_[i];
        const u64 lo = std::max(base, r.base), hi = std::min(base + size, r.base + r.size);
        if (r.base < lo) pieces.push_back({r.base, lo - r.base, r.perm, r.type, r.name});
        pieces.push_back({lo, hi - lo, perm, r.type, r.name});
        if (hi < r.base + r.size) pieces.push_back({hi, r.base + r.size - hi, r.perm, r.type, r.name});
    }
    const auto at = regions_.erase(regions_.begin() + static_cast<std::ptrdiff_t>(a),
                                   regions_.begin() + static_cast<std::ptrdiff_t>(b));
    regions_.insert(at, std::make_move_iterator(pieces.begin()), std::make_move_iterator(pieces.end()));
    RefreshPagesLocked(base, base + size);
    gen_.fetch_add(1, std::memory_order_release);
    return true;
}

void Memory::UnmapRange(u64 base, u64 size) {
    const u64 t0 = MapNow();
    std::unique_lock lock(mu_);
    const u64 t1 = MapNow();
    // In place, as Protect: erase the overlapped regions and put back what lies outside the range.
    const auto [a, b] = OverlapLocked(base, base + size);
    if (a < b) {
        Region pieces[2];
        int n = 0;
        if (const Region& f = regions_[a]; f.base < base) pieces[n++] = {f.base, base - f.base, f.perm, f.type, f.name};
        if (const Region& l = regions_[b - 1]; l.base + l.size > base + size) {
            pieces[n++] = {base + size, l.base + l.size - (base + size), l.perm, l.type, l.name};
        }
        const auto at = regions_.erase(regions_.begin() + static_cast<std::ptrdiff_t>(a),
                                       regions_.begin() + static_cast<std::ptrdiff_t>(b));
        regions_.insert(at, std::make_move_iterator(pieces), std::make_move_iterator(pieces + n));
    }
    const u64 t2 = MapNow();
    RefreshPagesLocked(base, base + size);
    gen_.fetch_add(1, std::memory_order_release);
    const u64 t3 = MapNow();
    ReleaseDeviceVisibleLocked(base, size);
    const u64 t4 = MapNow();
    // Give the host pages back: a later mapping must read as zero, as fresh pages do. (Zeroing on a helper
    // thread instead was tried 2026-10-21 and reverted: the game maps freed pages again soon, so maps had to
    // zero them themselves under the kernel lock, holding it 17-54 ms where this path holds it 5-6: README.)
    writetrack::Forget(base, size); // writable again before the zero-fill, and changed
    platform::ZeroWindow(Raw(base), size);
    if (MapStatsOn()) {
        UnmapStats& u = g_unmap_stats;
        u.lock += t1 - t0, u.regions += t2 - t1, u.refresh += t3 - t2, u.device += t4 - t3, u.zero += MapNow() - t4;
        u.bytes += size;
        const u64 n = ++u.n;
        if (n % 500 == 0) {
            auto ms = [](std::atomic<u64>& v) { return v.exchange(0) / 1e6; };
            const double l = ms(u.lock), r = ms(u.regions), p = ms(u.refresh), d = ms(u.device), z = ms(u.zero);
            Log("unmap: %llu unmaps; last 500 (%.1f MB): lock %.1f ms, regions %.1f ms, pages %.1f ms, device %.1f ms, "
                "zero %.1f ms; %zu regions",
                (unsigned long long)n, u.bytes.exchange(0) / 1048576.0, l, r, p, d, z, regions_.size());
        }
    }
}

u8* Memory::DeviceBacking() {
    if (!device_base_) {
        device_base_ = platform::CreateDeviceBacking(kDeviceBackingSize);
        if (!device_base_) Fatal("cannot create the device memory backing");
    }
    return device_base_;
}

void Memory::MakeDeviceVisible(u64 va, u64 size) {
    DeviceBacking();
    // Copy-then-remap below: a guest thread writing these pages at the same moment
    // could lose that write. They are freshly allocated GPU buffers in practice.
    std::unique_lock lock(mu_);
    const u64 end = (va + size + 0xFFF) & ~0xFFFULL;
    va &= ~0xFFFULL;
    // Walk [va, end) and back each gap between existing extents.
    auto it = device_extents_.upper_bound(va);
    if (it != device_extents_.begin() && std::prev(it)->first + std::prev(it)->second.size > va) --it;
    u64 at = va;
    while (at < end) {
        u64 gap_end = end;
        if (it != device_extents_.end() && it->first <= at) { // inside an existing extent
            at = it->first + it->second.size;
            ++it;
            continue;
        }
        if (it != device_extents_.end()) gap_end = std::min(end, it->first);
        const u64 len = gap_end - at;
        if (device_used_ + len > kDeviceBackingSize) Fatal("device memory backing exhausted");
        const u64 phys = device_used_;
        device_used_ += len;
        // Copy what the guest already has there, then put the backing in its place (never write-tracked from
        // now on: the backing's own view writes these pages without faulting).
        writetrack::Exclude(at, len, true);
        std::memcpy(device_base_ + phys, Raw(at), len);
        if (!platform::MapDeviceView(Raw(at), len, phys)) {
            Fatal("cannot alias 0x%llx+0x%llx onto the device backing", (unsigned long long)at,
                  (unsigned long long)len);
        }
        device_extents_[at] = {len, phys};
        at = gap_end;
    }
}

u8* Memory::DevicePointer(u64 va) const {
    std::shared_lock lock(mu_);
    auto it = device_extents_.upper_bound(va);
    if (it == device_extents_.begin()) return nullptr;
    --it;
    if (va - it->first >= it->second.size) return nullptr;
    return device_base_ + it->second.phys + (va - it->first);
}

void Memory::ReleaseDeviceVisibleLocked(u64 va, u64 size) {
    // Unmapped guest memory goes back to private pages. (Its backing range is
    // not reused: the allocator only grows.)
    const u64 end = va + size;
    auto it = device_extents_.upper_bound(va);
    if (it != device_extents_.begin()) --it;
    while (it != device_extents_.end() && it->first < end) {
        const u64 b = it->first, e = b + it->second.size;
        if (e <= va) {
            ++it;
            continue;
        }
        const DeviceExtent ext = it->second;
        it = device_extents_.erase(it);
        const u64 lo = std::max(b, va), hi = std::min(e, end);
        if (b < lo) device_extents_[b] = {lo - b, ext.phys};
        if (hi < e) device_extents_[hi] = {e - hi, ext.phys + (hi - b)};
        platform::UnmapDeviceView(Raw(lo), hi - lo);
        writetrack::Exclude(lo, hi - lo, false);
    }
}

void Memory::Unmap(u64 base) {
    std::unique_lock lock(mu_);
    u64 end = base;
    for (const auto& r : regions_) {
        if (r.base == base) end = std::max(end, r.base + r.size);
    }
    std::erase_if(regions_, [base](const Region& r) { return r.base == base; });
    writetrack::Forget(base, end - base);
    RefreshPagesLocked(base, end);
    gen_.fetch_add(1, std::memory_order_release);
}

// The returned region stays valid while the map does not change; callers outside
// guest code hold the kernel lock, which every change of the map also holds.
const Region* Memory::Find(u64 va) const {
    std::shared_lock lock(mu_);
    return FindLocked(va);
}

const Region* Memory::FindLocked(u64 va) const {
    // Regions are sorted by base and never overlap.
    auto it = std::upper_bound(regions_.begin(), regions_.end(), va,
                               [](u64 v, const Region& r) { return v < r.base; });
    if (it == regions_.begin()) {
        return nullptr;
    }
    --it;
    if (va - it->base >= it->size) {
        return nullptr;
    }
    return &*it;
}

u8* Memory::Access(u64 va, u64 size, u32 perm) {
    va &= 0xFFFF'FFFF'FFFFULL; // AArch64 ignores the top byte for data accesses (TBI)
    // Each thread caches the last region it used, valid until the map changes
    // (gen_), so most accesses take no lock.
    thread_local struct {
        u64 gen = 0, base = 0, size = 0;
        u32 perm = 0;
    } cached;
    const u64 gen = gen_.load(std::memory_order_acquire);
    if (cached.gen != gen || va - cached.base >= cached.size || va + size > cached.base + cached.size) {
        std::shared_lock lock(mu_);
        const Region* r = FindLocked(va);
        if (!r || va + size > r->base + r->size) {
            Fatal("%s of %llu bytes at unmapped 0x%llx", perm & kWrite ? "write" : "read",
                  (unsigned long long)size, (unsigned long long)va);
        }
        if ((r->perm & perm) != perm) {
            Fatal("%s of %llu bytes at 0x%llx violates %s permissions",
                  perm & kWrite ? "write" : "read", (unsigned long long)size,
                  (unsigned long long)va, r->name.c_str());
        }
        cached = {gen, r->base, r->size, r->perm};
    } else if ((cached.perm & perm) != perm) {
        std::shared_lock lock(mu_);
        const Region* r = FindLocked(va);
        Fatal("%s of %llu bytes at 0x%llx violates %s permissions", perm & kWrite ? "write" : "read",
              (unsigned long long)size, (unsigned long long)va, r ? r->name.c_str() : "?");
    }
    if (perm & kWrite) writetrack::HostWrite(va, size); // the caller may hand it to the OS kernel: no fault there
    if (IsGpuCached(va, size)) {
        if (perm & kWrite) {
            gpu::OnCpuWrite(va, size);
        } else {
            gpu::OnCpuRead(va, size);
        }
    }
    return Raw(va);
}

void Memory::RefreshPagesLocked(u64 lo, u64 hi) {
    if (!pages_) return;
    lo &= ~0xFFFULL;
    hi = std::min<u64>((hi + 0xFFF) & ~0xFFFULL, kAddressSpaceEnd);
    for (u64 page = lo >> 12; page < hi >> 12; ++page) pages_[page].store(0, std::memory_order_relaxed);
    const uintptr_t host = reinterpret_cast<uintptr_t>(host_);
    const auto [first, last] = OverlapLocked(lo, hi);
    for (size_t i = first; i < last; ++i) {
        const Region& r = regions_[i];
        if (!(r.perm & kRead)) continue; // no inline access: the callback reports the fault
        const u64 b = std::max(lo, r.base), e = std::min(hi, r.base + r.size);
        for (u64 va = b & ~0xFFFULL; va < e; va += 0x1000) {
            pages_[va >> 12].store(host | (IsGpuCached(va, 1) ? kPageGpuCached : kPageMemory), std::memory_order_release);
        }
    }
}

void Memory::EnableFastmem(RecompHostMem& hm) {
    std::unique_lock lock(mu_);
    if (reinterpret_cast<uintptr_t>(host_) & kPageTagMask) Fatal("guest window not page aligned");
    const size_t bytes = (kAddressSpaceEnd >> 12) * sizeof(uintptr_t); // 1 GB, touched sparsely
    void* p = platform::AllocateSparseZeroed(bytes);
    if (!p) Fatal("cannot reserve the fastmem page table");
    pages_ = static_cast<std::atomic<uintptr_t>*>(p);
    const uintptr_t host = reinterpret_cast<uintptr_t>(host_);
    for (const auto& r : regions_) { // a fresh table is all zero: only mapped pages need writing
        if (!(r.perm & kRead)) continue;
        for (u64 va = r.base & ~0xFFFULL; va < r.base + r.size; va += 0x1000) {
            pages_[va >> 12].store(host | (IsGpuCached(va, 1) ? kPageGpuCached : kPageMemory), std::memory_order_relaxed);
        }
    }
    hm.page_entries = pages_;
    hm.page_entry_stride = sizeof(uintptr_t);
    hm.page_bits = 12;
    hm.pointer_mask = ~kPageTagMask;
    hm.address_space_max = kAddressSpaceEnd;
    hm.guard_generation = nullptr;
    hm.page_type_mask = kPageTagMask;
    hm.page_type_memory = kPageMemory;
    hm.page_type_debug = kPageDebug;
    hm.page_type_rasterizer_cached = kPageGpuCached;
    hm.counters = nullptr; // a shared atomic per access would cost more than it tells
}

void Memory::MarkGpuCached(u64 va, u64 size, bool cached) {
    for (u64 page = va >> 12, end = (va + size + 0xFFF) >> 12; page < end; ++page) {
        auto& slot = gpu_cached_[page >> kGpuChunkBits];
        GpuChunk* chunk = slot.load(std::memory_order_acquire);
        if (!chunk) {
            if (!cached) continue;
            auto* fresh = new GpuChunk{};
            if (slot.compare_exchange_strong(chunk, fresh, std::memory_order_acq_rel)) {
                chunk = fresh;
            } else {
                delete fresh; // another thread installed one; chunk now holds it
            }
        }
        auto& count = chunk->count[page & ((1u << kGpuChunkBits) - 1)];
        u16 before = 0;
        if (cached) {
            before = count.fetch_add(1, std::memory_order_relaxed);
        } else if (count.load(std::memory_order_relaxed) > 0) {
            before = count.fetch_sub(1, std::memory_order_relaxed);
        } else {
            continue;
        }
        if (cached && before == 0 && g_direct_mem) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                Log("memory: WARNING: video_core caches guest page 0x%llx, but the direct lift does not report CPU "
                    "writes to it (old_plans/FASTMEM_PLAN.md Phase 4)", (unsigned long long)(page << 12));
            }
        }
        // Fastmem: a page entering or leaving the GPU caches switches between the
        // inline path and the callbacks (which tell video_core about the access).
        if (pages_ && (cached ? before == 0 : before == 1)) {
            auto& e = pages_[page];
            uintptr_t v = e.load(std::memory_order_relaxed);
            const uintptr_t from = cached ? kPageMemory : kPageGpuCached, to = cached ? kPageGpuCached : kPageMemory;
            while ((v & kPageTagMask) == from &&
                   !e.compare_exchange_weak(v, (v & ~kPageTagMask) | to, std::memory_order_release)) {
            }
        }
    }
}

bool Memory::IsGpuCached(u64 va, u64 size) const {
    for (u64 page = va >> 12, end = (va + size + 0xFFF) >> 12; page < end; ++page) {
        const GpuChunk* chunk = gpu_cached_[page >> kGpuChunkBits].load(std::memory_order_acquire);
        if (!chunk) {
            page |= (1u << kGpuChunkBits) - 1; // skip the rest of this chunk
            continue;
        }
        if (chunk->count[page & ((1u << kGpuChunkBits) - 1)].load(std::memory_order_relaxed)) return true;
    }
    return false;
}

// Zero bytes touch no memory, wherever `va` points (games pass a count of 0 with a null array).
void Memory::Write(u64 va, const void* data, u64 size) {
    if (size) std::memcpy(Access(va, size, kWrite), data, size);
}

void Memory::Read(u64 va, void* data, u64 size) {
    if (size) std::memcpy(data, Access(va, size, kRead), size);
}

std::string Memory::ReadString(u64 va, u64 max) {
    std::string s;
    for (u64 n = 0; n < max; ++n) {
        const char ch = static_cast<char>(*Access(va + n, 1, kRead));
        if (!ch) {
            break;
        }
        s.push_back(ch);
    }
    return s;
}

// --- Diagnostics -------------------------------------------------------------

static const auto g_uptime_start = std::chrono::steady_clock::now(); // set at program load

double Uptime() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_uptime_start).count();
}

std::string Describe(u64 va) {
    char b[160];
    if (hle::IsHostAddress(va)) {
        std::snprintf(b, sizeof b, "0x%llx (host:%s)", (unsigned long long)va, hle::HostFunctionName(va).c_str());
        return b;
    }
    if (va >= kImageBase && va < kImageBase + rt_image_size) {
        const u64 off = va - kImageBase;
        auto it = g_symbols.upper_bound(off);
        if (it != g_symbols.begin()) {
            --it;
            std::snprintf(b, sizeof b, "0x%llx (%s+0x%llx)", (unsigned long long)va,
                          it->second.c_str(), (unsigned long long)(off - it->first));
            return b;
        }
    }
    std::snprintf(b, sizeof b, "0x%llx", (unsigned long long)va);
    return b;
}

void Log(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    VLog("[rt] ", fmt, ap);
    va_end(ap);
}

void Fatal(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    VLog("[rt] FATAL: ", fmt, ap);
    va_end(ap);
    LogGuestState();
    std::fflush(stderr);
    std::_Exit(70);
}

void LogGuestState() {
    const GuestContext* c = CurrentContext();
    const KThread* t = CurrentThread();
    std::fprintf(stderr, "[rt]   thread %s, guest pc %s, lr %s, sp 0x%llx\n", t ? t->name.c_str() : "(not a guest thread)",
                 Describe(kImageBase + rt_trace_pc).c_str(), c ? Describe(c->x[30]).c_str() : "?",
                 c ? (unsigned long long)c->x[31] : 0ULL);
}

void Exit(int code) {
    FinishCaptures();
    nvn::SavePipelineCacheAtExit();
    std::fflush(stdout);
    Log("guest exited with code %d", code);
    std::fflush(stderr);
    std::_Exit(code); // other guest threads may still be running or blocked
}

} // namespace rt

using namespace rt;

// --- Called from generated code ----------------------------------------------

extern "C" BlockFn recomp_lookup(uint64_t pc) {
    if (hle::IsHostAddress(pc) || hle::IsOverridden(pc)) {
        return [](GuestContext* c) { rt_jump(c, c->pc); }; // host functions and overrides: rt_jump's path
    }
    const RtFunc* f = FindFunc(pc);
    return f ? f->fn : nullptr;
}

static bool InImageText(u64 va) {
    for (unsigned long i = 0; i < rt_segment_count; ++i) {
        const RtSegment& s = rt_segments[i];
        if ((s.perm & 4) && va >= kImageBase + s.offset && va < kImageBase + s.offset + s.size) {
            return true;
        }
    }
    return false;
}

extern "C" void rt_jump(GuestContext* c, uint64_t target) {
    if (target == kLoaderReturn) {
        Exit(static_cast<int>(c->x[0]));
    }
    if (target == kThreadExitReturn) {
        ExitCurrentThread();
    }
    MaybePreempt();
    if (hle::IsHostAddress(target)) {
        hle::CallHost(c, target);
        return;
    }
    JumpCache* jc = t_jump_cache;
    const u64 gen = g_jump_gen.load(std::memory_order_acquire);
    if (!jc || jc->gen != gen) {
        if (!jc) jc = t_jump_cache = new JumpCache; // kept for the thread's life (guest threads are few)
        std::memset(jc->e, 0, sizeof jc->e);
        jc->gen = gen;
    }
    JumpCache::Entry& e = jc->e[(target >> 2) & (JumpCache::kEntries - 1)];
    if (e.target == target && e.f) {
        const RtFunc* hit = e.f;
        hit->fn(c);
        return;
    }
    // Overrides replace lifted function starts only (hle::OverrideGuestFunction), marked as such.
    const RtFunc* f = FindFunc(target);
    if (f && Overridden(f)) {
        if (hle::MaybeCallOverride(c, target)) return;
    } else if (f) {
        e = {target, f};
    }
    if (!f) {
        // A jump into the middle of a function, or outside the image: record it so
        // the next lift can seed it (nxrecomp --extra-seeds gaps.txt).
        if (InImageText(target)) {
            if (FILE* g = std::fopen("gaps.txt", "a")) {
                std::fprintf(g, "%llx\n", (unsigned long long)(target - kImageBase));
                std::fclose(g);
            }
        }
        Fatal("indirect branch to %s, which is not a lifted function start", Describe(target).c_str());
    }
    f->fn(c);
}

extern "C" void rt_break(GuestContext*, uint32_t imm, uint64_t pc) {
    Fatal("brk #0x%x at %s", imm, Describe(pc).c_str());
}

/// The guest's main thread, from the entry point on (its own host thread).
[[noreturn]] static void RunGuestMain(u64 stack_base) {
    // NSO-style entry: x0 = 0, x1 = main thread handle.
    // The main thread holds the global lock while it runs guest code only in
    // the global-lock mode (threads.h).
    GilLock gil(g_gil);
    auto main_thread = StartMainThread(gil);
    Log("threads: %s", g_parallel ? "parallel (kernel lock only in SVCs)" : "one global lock (RT_GIL=1)");
    GuestContext* c = &main_thread->ctx;
    c->host_mem = &g_host_mem;
    c->pending_svc = ~0ULL;
    c->x[0] = 0;
    c->x[1] = kMainThreadHandle;
    c->x[30] = kLoaderReturn;
    c->x[31] = stack_base + kMainStackSize;
    c->tpidrro_el0 = kTlsBase;
    c->pc = kImageBase + rt_entry;
    Log("image 0x%llx bytes at 0x%llx, %lu functions, entry %s",
        (unsigned long long)rt_image_size, (unsigned long long)kImageBase, rt_func_count,
        Describe(c->pc).c_str());

    if (g_parallel) gil.unlock();
    // Top-level dispatch: generated functions return when the guest returns to an
    // address none of their callers is waiting for (see the lifter's ResumeCheck).
    for (;;) {
        rt_jump(c, c->pc);
    }
}

// --- Startup -------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <image.bin> [symbols.txt]\n", argv[0]);
        return 2;
    }
    platform::InitProcess();
    // The guest window first, at its fixed host address, before anything else in the process allocates
    // (threads, the window, the GPU driver): a direct lift needs exactly that address (rt.h kHostWindowBase).
    g_mem.Init();
    // The log is buffered and written by a thread of its own every 100 ms: unbuffered, every line was three
    // writes made with the kernel lock held (SVC traces), which on Windows, to a file on \\wsl.localhost, made
    // the title load in 58 s instead of 23 and stalled frames for ~0.5 s (2026-10-18). Fatal, crashes and
    // Exit flush it; a killed run loses at most the last 100 ms.
    static char log_buffer[1 << 20];
    std::setvbuf(stderr, log_buffer, _IOFBF, sizeof log_buffer);
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::fflush(stderr);
        }
    }).detach();
    platform::InstallCrashReport(LogGuestState);
    writetrack::Init();
    if (const char* e = std::getenv("RT_WRITE_TRACK_SELFTEST"); e && *e && *e != '0') {
        const bool ok = writetrack::SelfTest();
        std::fflush(stderr);
        std::exit(ok ? 0 : 1);
    }
    if (argc > 2) {
        LoadSymbols(argv[2]);
    }

    diag::Start();
    options::LogAll();
    window::Open(); // early: a missing display fails before the image loads
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> image((std::istreambuf_iterator<char>(f)), {});
    if (image.size() != rt_image_size) {
        Fatal("%s is 0x%zx bytes, but the lifted code expects 0x%llx", argv[1], image.size(),
              (unsigned long long)rt_image_size);
    }
    if (rt_load_base != 0 && rt_load_base != kImageBase) {
        Fatal("the image was linked at 0x%llx, but the runtime loads it at 0x%llx",
              (unsigned long long)rt_load_base, (unsigned long long)kImageBase);
    }
    // Segments as the lifter describes them. A self-relocating image (libnx) has
    // writable rodata so it can apply its own relocations.
    u64 mapped_end = 0; // segments sharing a page: the first one's permissions win
    for (unsigned long i = 0; i < rt_segment_count; ++i) {
        const RtSegment& s = rt_segments[i];
        const u64 begin = std::max<u64>(s.offset & ~0xFFFULL, mapped_end);
        const u64 end = (s.offset + s.size + 0xFFF) & ~0xFFFULL;
        if (end <= begin) {
            continue;
        }
        mapped_end = end;
        const u32 perm = (s.perm & 1 ? u32{kRead} : 0) | (s.perm & 2 ? u32{kWrite} : 0) | (s.perm & 4 ? u32{kExec} : 0);
        const MemType type = s.perm & 2 ? kMemCodeMutable : kMemCodeStatic;
        g_mem.Map(kImageBase + begin, end - begin, perm, type,
                  s.perm & 4 ? "image .text" : s.perm & 2 ? "image data" : "image rodata");
    }
    std::memcpy(g_mem.Raw(kImageBase), image.data(), image.size());

    const u64 stack_base = kStackRegion;
    g_mem.Map(stack_base, kMainStackSize, kRead | kWrite, kMemStack, "main stack");
    g_mem.Map(kTlsBase, 0x1000, kRead | kWrite, kMemThreadLocal, "main TLS");
    // Host functions and overrides; slots.txt lies next to the image (nxrecomp's output).
    {
        // (std::filesystem: Windows paths separate with '\\', \\wsl.localhost\... ones too)
        const std::filesystem::path dir = std::filesystem::path(argv[1]).parent_path();
        hle::Init(((dir.empty() ? std::filesystem::path(".") : dir) / "slots.txt").string());
    }
    nvn::Init();
    nvn::tap::Init();

    g_host_mem = {};
    g_host_mem.load = Load;
    g_host_mem.store = Store;
    g_host_mem.excl_load = ExclLoad;
    g_host_mem.excl_store = ExclStore;
    g_host_mem.clear_excl = ClearExcl;
    g_host_mem.read_cntpct = ReadCntpct;
    g_host_mem.excl_load_pair = ExclLoadPair;
    g_host_mem.excl_store_pair = ExclStorePair;
    // Fastmem needs parallel threads: under the global lock a spin-wait on the
    // inline path would never reach a preemption point (threads.h).
    const bool fastmem = g_parallel && !std::getenv("RT_NO_FASTMEM");
    if (const u64 base = LiftDirectBase()) {
        // The lifted code goes straight to the window: no callbacks, no preemption point, no GPU write tracking.
        if (base != kHostWindowBase) {
            Fatal("the lift expects guest memory at host 0x%llx, the runtime puts it at 0x%llx: re-lift",
                  (unsigned long long)base, (unsigned long long)kHostWindowBase);
        }
        const char* why = !g_parallel ? "RT_GIL=1 (a spin-wait would never reach a preemption point)"
                          : !fastmem  ? "RT_NO_FASTMEM"
                          : !nvn::Enabled() ? "the driver seam (video_core needs to see CPU writes; use RT_GPU=nvn)"
                                            : nullptr;
        if (why) Fatal("this lift (direct memory, the nxrecomp default) does not support %s: re-lift with --no-direct-mem", why);
        g_direct_mem = true;
    }
    if (fastmem) g_mem.EnableFastmem(g_host_mem);
    Log("memory: %s", g_direct_mem ? "direct (lifted code reaches the window itself; exclusives through the helpers)"
                      : fastmem    ? "fastmem (plain pages inline, GPU-cached pages through the callbacks)"
                                   : "every access through the callbacks");
    g_module_base = kImageBase;

    // The guest's main thread gets a host thread of its own; this one runs the
    // window's event loop (window.h).
    std::thread(RunGuestMain, stack_base).detach();
    window::RunEventLoop();
}
