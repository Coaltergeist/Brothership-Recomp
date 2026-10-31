// CPU write tracking for guest memory: see writetrack.h.
#include "writetrack.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

#include "platform/platform.h"

namespace rt::writetrack {
namespace {

constexpr u64 kPage = 0x1000;
constexpr u64 kChunkBits = 21; // the page table is allocated in pieces covering 2 MB of guest memory
constexpr u64 kChunkPages = (u64{1} << kChunkBits) / kPage;
constexpr u32 kHotAfter = 3, kHotFor = 60; // the hot-page rule (writetrack.h)

enum : u8 { kFree, kWatched, kReleasing };

struct Page {
    std::atomic<u32> gen{0};        // the generation of the last write seen
    std::atomic<u32> watched_at{0}; // the generation its last protection began (0: never; a new one per Watch call)
    std::atomic<u8> state{kFree};
    std::atomic<u8> excluded{0};    // also mapped elsewhere (Exclude)
    u8 streak = 0, cooldown = 0;    // the hot-page rule
    u64 hash = 0;                   // RT_WRITE_TRACK_CHECK: the bytes when protected
};

std::atomic<Page*> g_chunks[kAddressSpaceEnd >> kChunkBits];
std::atomic<u32> g_now{1};
std::atomic<bool> g_any{false}; // anything was ever watched (HostWrite and Forget cost nothing before)
uintptr_t g_host = 0;           // the guest window's host address

struct Stats {
    std::atomic<u64> faults{0}, host_writes{0}, watched{0}, hot{0}, missed{0}, protect_ns{0}, failed{0};
} g_stats;

bool StatsOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_WRITE_TRACK_STATS");
        return e && *e && *e != '0';
    }();
    return on;
}

u64 NowNs() {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

Page* PageAt(u64 va, bool create) {
    if (va >= kAddressSpaceEnd) return nullptr;
    std::atomic<Page*>& slot = g_chunks[va >> kChunkBits];
    Page* chunk = slot.load(std::memory_order_acquire);
    if (!chunk && create) {
        auto* fresh = new Page[kChunkPages];
        if (slot.compare_exchange_strong(chunk, fresh, std::memory_order_acq_rel)) chunk = fresh;
        else delete[] fresh; // another thread's won (chunk now holds it)
    }
    return chunk ? &chunk[(va & ((u64{1} << kChunkBits) - 1)) / kPage] : nullptr;
}

/// Calls f(page va, Page&) for the pages of [lo, hi) whose piece of the table exists (or is created).
template <typename F>
void ForPages(u64 lo, u64 hi, bool create, F f) {
    lo &= ~(kPage - 1);
    hi = std::min<u64>((hi + kPage - 1) & ~(kPage - 1), kAddressSpaceEnd);
    for (u64 va = lo; va < hi;) {
        Page* p = PageAt(va, create);
        if (!p) { // skip the piece
            va = ((va >> kChunkBits) + 1) << kChunkBits;
            continue;
        }
        f(va, *p);
        va += kPage;
    }
}

u64 PageHash(u64 va) {
    const u8* d = g_mem.Raw(va);
    u64 h = 0x9e3779b97f4a7c15ull;
    for (u64 k = 0; k < kPage; k += 8) {
        u64 w;
        std::memcpy(&w, d + k, 8);
        h = (h ^ w) * 0xff51afd7ed558ccdull, h ^= h >> 32;
    }
    return h;
}

void Bump(Page& p) { p.gen.store(g_now.fetch_add(1, std::memory_order_acq_rel) + 1, std::memory_order_release); }

/// A watched page: unprotect it and bump its generation (true). Another thread doing so: wait for it (true).
/// Not watched: false.
bool Release(u64 va, Page& p) {
    u8 s = kWatched;
    if (p.state.compare_exchange_strong(s, kReleasing, std::memory_order_acq_rel)) {
        Bump(p);
        platform::ProtectReadWrite(g_mem.Raw(va), kPage);
        p.state.store(kFree, std::memory_order_release);
        return true;
    }
    if (s == kReleasing) {
        while (p.state.load(std::memory_order_acquire) == kReleasing) std::this_thread::yield();
        return true;
    }
    return false;
}

bool OnWriteFault(uintptr_t a) {
    if (!g_any.load(std::memory_order_relaxed) || a < g_host) return false;
    const u64 va = (a - g_host) & ~(kPage - 1);
    Page* p = PageAt(va, false);
    if (!p) return false;
    if (Release(va, *p)) {
        g_stats.faults.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    // Not watched (any more): another thread may have released it between the fault and here - retry, unless the
    // same address keeps faulting (then it is not ours: the remap handler or the crash report)
    if (!p->watched_at.load(std::memory_order_relaxed)) return false;
    thread_local uintptr_t last = 0;
    thread_local unsigned repeats = 0;
    repeats = a == last ? repeats + 1 : 0;
    last = a;
    return repeats < 1000;
}

void MaybeLogStats() {
    if (!StatsOn()) return;
    static std::atomic<double> next{5};
    double at = next.load(std::memory_order_relaxed);
    const double now = Uptime();
    if (now < at || !next.compare_exchange_strong(at, now + 5)) return;
    Log("writetrack: last 5 s: %llu write faults, %llu host writes released, %llu pages protected (%.1f ms protecting), "
        "%llu skipped as hot, %llu could not be protected; missed writes found %llu (RT_WRITE_TRACK_CHECK)",
        (unsigned long long)g_stats.faults.exchange(0), (unsigned long long)g_stats.host_writes.exchange(0),
        (unsigned long long)g_stats.watched.exchange(0), g_stats.protect_ns.exchange(0) / 1e6,
        (unsigned long long)g_stats.hot.exchange(0), (unsigned long long)g_stats.failed.exchange(0),
        (unsigned long long)g_stats.missed.load());
}

} // namespace

bool On() {
    static const bool on = [] {
        const char* e = std::getenv("RT_WRITE_TRACK");
        return !(e && *e == '0');
    }();
    return on;
}

u32 Now() { return g_now.load(std::memory_order_acquire); }

bool CheckOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_WRITE_TRACK_CHECK");
        return e && *e && *e != '0';
    }();
    return on;
}

std::string Describe(u64 va) {
    const Page* p = PageAt(va & ~(kPage - 1), false);
    if (!p) return "untracked";
    const u8 s = p->state.load();
    char b[128];
    std::snprintf(b, sizeof b, "%s gen %u watched_at %u streak %u%s", s == kWatched ? "watched" : s == kFree ? "free" : "releasing",
                  p->gen.load(), p->watched_at.load(), p->streak, p->excluded.load() ? " excluded" : "");
    return b;
}

void Init() {
    g_host = reinterpret_cast<uintptr_t>(g_mem.Raw(0));
    platform::SetWriteFaultHook(OnWriteFault);
}

void Watch(u64 va, u64 size) {
    if (!On() || !size) return;
    g_any.store(true, std::memory_order_relaxed);
    // pages protected by this call start a generation of their own: a `since` taken before it is older (the page
    // may have been written while unprotected), one taken after it is not
    const u32 now = g_now.fetch_add(1, std::memory_order_acq_rel) + 1;
    const bool check = CheckOn();
    thread_local std::vector<std::array<u64, 2>> runs; // [lo, hi) to protect
    runs.clear();
    // Under the map's shared lock: no remap can change these pages meanwhile; only read-write guest memory
    g_mem.ForEachOverlapping(va, va + size, [&](const Region& r) {
        if (!(r.perm & kWrite)) return;
        const u64 lo = std::max(va, r.base), hi = std::min(va + size, r.base + r.size);
        ForPages(lo, hi, true, [&](u64 page, Page& p) {
            if (page < r.base || page + kPage > r.base + r.size) return; // a page the region shares with another
            if (p.excluded.load(std::memory_order_relaxed)) return;
            const u8 s = p.state.load(std::memory_order_acquire);
            if (s == kWatched) { // no write since it was protected
                p.streak = 0;
                if (check && PageHash(page) != p.hash) {
                    const u64 n = g_stats.missed.fetch_add(1) + 1;
                    if (n <= 20 || n % 1000 == 0) {
                        Log("writetrack: MISSED WRITE #%llu: page 0x%llx changed while watched (generation %u)",
                            (unsigned long long)n, (unsigned long long)page, p.gen.load());
                    }
                    p.hash = PageHash(page);
                }
                return;
            }
            if (s == kReleasing) return;
            const u32 w = p.watched_at.load(std::memory_order_relaxed);
            if (w && p.gen.load(std::memory_order_relaxed) > w) p.streak = static_cast<u8>(std::min(p.streak + 1, 255));
            if (p.streak >= kHotAfter) {
                if (++p.cooldown < kHotFor) {
                    g_stats.hot.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                p.streak = 0, p.cooldown = 0;
            }
            p.watched_at.store(now, std::memory_order_relaxed);
            p.state.store(kWatched, std::memory_order_release);
            if (!runs.empty() && runs.back()[1] == page) runs.back()[1] += kPage;
            else runs.push_back({page, page + kPage});
        });
        const u64 t0 = NowNs();
        for (const auto& [lo2, hi2] : runs) {
            u64 n = (hi2 - lo2) / kPage;
            if (!platform::ProtectReadOnly(g_mem.Raw(lo2), hi2 - lo2)) { // across separately mapped views: page by page
                n = 0;
                for (u64 page = lo2; page < hi2; page += kPage) {
                    if (platform::ProtectReadOnly(g_mem.Raw(page), kPage)) {
                        ++n;
                        continue;
                    }
                    PageAt(page, false)->state.store(kFree, std::memory_order_release); // stays unwatched: "changed"
                    g_stats.failed.fetch_add(1, std::memory_order_relaxed);
                }
            }
            g_stats.watched.fetch_add(n, std::memory_order_relaxed);
            if (check) {
                for (u64 page = lo2; page < hi2; page += kPage) PageAt(page, false)->hash = PageHash(page);
            }
        }
        g_stats.protect_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
        runs.clear();
    });
    MaybeLogStats();
}

bool ChangedSince(u64 va, u64 size, u32 since) {
    if (!On() || !size) return true;
    const u64 lo = va & ~(kPage - 1), hi = (va + size + kPage - 1) & ~(kPage - 1);
    for (u64 page = lo; page < hi; page += kPage) {
        const Page* p = PageAt(page, false);
        if (!p || p->state.load(std::memory_order_acquire) != kWatched || p->gen.load(std::memory_order_acquire) > since ||
            p->watched_at.load(std::memory_order_acquire) > since) {
            return true;
        }
    }
    return false;
}

void HostWrite(u64 va, u64 size) {
    if (!g_any.load(std::memory_order_relaxed) || !size) return;
    ForPages(va, va + size, false, [](u64 page, Page& p) {
        if (Release(page, p)) g_stats.host_writes.fetch_add(1, std::memory_order_relaxed);
    });
}

void Forget(u64 va, u64 size) {
    if (!g_any.load(std::memory_order_relaxed) || !size) return;
    ForPages(va, va + size, false, [](u64 page, Page& p) {
        if (!Release(page, p)) Bump(p);
        p.streak = 0, p.cooldown = 0;
    });
}

void Exclude(u64 va, u64 size, bool excluded) {
    ForPages(va, va + size, excluded, [&](u64 page, Page& p) {
        if (excluded) Release(page, p);
        else Bump(p);
        p.excluded.store(excluded ? 1 : 0, std::memory_order_release);
    });
}

bool SelfTest() {
    constexpr u64 kBase = 0x7F'0000'0000, kPages = 64; // a free spot near the top of the 39-bit space
    bool ok = true;
    const auto expect = [&](bool cond, const char* what) {
        Log("writetrack selftest: %s: %s", cond ? "ok  " : "FAIL", what);
        ok = ok && cond;
    };
    const auto pg = [](u64 i) { return kBase + i * kPage; };
    g_mem.Map(kBase, kPages * kPage, kRead | kWrite, kMemNormal, "writetrack selftest");
    std::memset(g_mem.Raw(kBase), 0x11, kPages * kPage);

    // 1. host code writes (a plain store into the window: the same fault lifted code takes)
    Watch(kBase, 32 * kPage);
    u32 since = Now();
    expect(!ChangedSince(kBase, 32 * kPage, since), "a watched range with no write is unchanged");
    *g_mem.Raw(pg(1)) = 0x22;
    expect(ChangedSince(pg(1), kPage, since), "a store into a watched page faults and is seen");
    expect(*g_mem.Raw(pg(1)) == 0x22, "the store reached memory after the fault");
    expect(!ChangedSince(pg(2), kPage, since), "the page next to it is still unchanged");
    volatile u8 read = *g_mem.Raw(pg(2));
    (void)read;
    expect(!ChangedSince(pg(2), kPage, since), "a read does not count as a write");

    // 2. host code through Memory::Access(kWrite) (HostWrite: no fault at all)
    std::memset(g_mem.Access(pg(3), 16, kWrite), 0x33, 16);
    expect(ChangedSince(pg(3), kPage, since), "a write through Memory::Access(kWrite) is seen");

    // 3. the OS kernel writing guest memory: a file read into a watched page
    const auto path = std::filesystem::temp_directory_path() / "nxrt_writetrack_selftest.bin";
    {
        std::vector<u8> data(kPage, 0x44);
        platform::File f;
        f.Open(path, platform::File::Mode::kCreate);
        f.WriteAt(data.data(), data.size(), 0);
    }
    platform::File f;
    f.Open(path, platform::File::Mode::kRead);
    const int64_t raw = f.ReadAt(g_mem.Raw(pg(5)), kPage, 0); // into a protected page, unannounced
    expect(raw < 0 || *g_mem.Raw(pg(5)) != 0x44, "(control) a kernel read into a protected page without HostWrite fails");
    const int64_t got = f.ReadAt(g_mem.Access(pg(4), kPage, kWrite), kPage, 0);
    expect(got == static_cast<int64_t>(kPage) && *g_mem.Raw(pg(4) + 100) == 0x44,
           "a file read through Memory::Access(kWrite) into a watched page works");
    expect(ChangedSince(pg(4), kPage, since), "and is seen");
    f.Close();
    std::filesystem::remove(path);

    // 4. many threads writing watched pages at once, some the same page
    Watch(pg(8), 24 * kPage);
    since = Now();
    std::vector<std::thread> threads;
    for (u64 k = 0; k < 8; ++k) {
        threads.emplace_back([k, &pg] {
            for (int round = 0; round < 3; ++round) {
                for (u64 i = 8 + k; i < 32; i += 8) *g_mem.Raw(pg(i) + k) = static_cast<u8>(round);
                *g_mem.Raw(pg(31) + 64 + k) = static_cast<u8>(round); // all eight on one page
            }
        });
    }
    for (auto& t : threads) t.join();
    bool all = true;
    for (u64 i = 8; i < 32; ++i) all = all && ChangedSince(pg(i), kPage, since);
    expect(all, "eight threads writing 24 watched pages (one shared by all) are all seen, none crashed");

    // 5. watched again: unchanged until the next write
    Watch(kBase, 32 * kPage);
    since = Now();
    expect(!ChangedSince(kBase, 32 * kPage, since), "after watching again, nothing counts as changed");

    // 6. the hot-page rule: written in each of 3 cycles, the 4th Watch leaves it writable ("always changed")
    for (int cycle = 0; cycle < 3; ++cycle) {
        Watch(pg(40), kPage);
        *g_mem.Raw(pg(40)) = static_cast<u8>(cycle);
    }
    Watch(pg(40), kPage);
    since = Now();
    expect(ChangedSince(pg(40), kPage, since), "a page written every cycle becomes hot (counts as changed, no fault)");
    // ...and a write while it is unprotected, then a later Watch that protects it again: changed since before then
    *g_mem.Raw(pg(40)) = 0x77; // no fault: hot
    for (u32 k = 0; k < kHotFor; ++k) Watch(pg(40), kPage); // until the rule tries it again and protects it
    expect(!ChangedSince(pg(40), kPage, Now()) && ChangedSince(pg(40), kPage, since),
           "a page written while unprotected and protected again counts as changed since before (the Phase 2 miss)");

    // 7. excluded (also mapped elsewhere): never watched
    Exclude(pg(42), kPage, true);
    Watch(pg(42), kPage);
    since = Now();
    expect(ChangedSince(pg(42), kPage, since), "an excluded page is never watched");
    Exclude(pg(42), kPage, false);

    // 8. given back: everything watched there counts as changed, and the pages are writable again
    Watch(pg(48), 8 * kPage);
    since = Now();
    g_mem.UnmapRange(kBase, kPages * kPage);
    expect(ChangedSince(pg(48), 8 * kPage, since), "unmapping watched pages counts as a write");
    g_mem.Map(kBase, kPages * kPage, kRead | kWrite, kMemNormal, "writetrack selftest");
    *g_mem.Raw(pg(48)) = 0x55;
    expect(*g_mem.Raw(pg(48)) == 0x55 && *g_mem.Raw(pg(49)) == 0, "mapped again, the pages read as zero and are writable");
    g_mem.UnmapRange(kBase, kPages * kPage);

    Log("writetrack selftest: %s (%llu write faults handled)", ok ? "PASS" : "FAIL",
        (unsigned long long)g_stats.faults.load());
    return ok;
}

} // namespace rt::writetrack
