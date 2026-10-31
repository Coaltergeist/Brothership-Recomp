// Horizon SVC handling. Results go in x0; outputs in x1.. as the libnx/SDK
// wrappers expect. Everything is logged so a run can be compared with an
// emulator's SVC trace.
#include <algorithm>
#include <array>
#include <map>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <random>
#include <thread>

#include "diag.h"
#include "ipc.h"
#include "kernel.h"
#include "threads.h"

namespace rt {
namespace {

constexpr u32 kInvalidEnumValue = 0xF001;   // kernel module 1, description 120
constexpr u32 kTimedOut = 0xEA01;           // kernel module 1, description 117
constexpr u32 kInvalidHandle = 0xE401;      // kernel module 1, description 114
constexpr u32 kInvalidState = 0xFA01;       // 125
constexpr u32 kInvalidCurrentMemory = 0xD401;

constexpr u64 kProcessId = 0x51;
constexpr u64 kProgramId = 0x0100'0000'0000'1000;

u64 g_heap_size = 0;
u64 g_physical_mapped = 0; // MapPhysicalMemory'd bytes (nn::os virtual address memory)

u64 UsedMemory() {
    u64 used = g_heap_size + g_physical_mapped;
    for (const auto& r : g_mem.Regions()) {
        if (r.type != kMemHeap && r.type != kMemNormal) {
            used += r.size;
        }
    }
    return used;
}

void QueryMemory(GuestContext* c) {
    // MemoryInfo { u64 addr, size; u32 type, attr, perm, ipc_refcount, device_refcount, pad; }
    const u64 out = c->x[0], va = c->x[2];
    u64 base = 0, size = 0;
    u32 type = kMemUnmapped, perm = 0;
    if (va >= kAddressSpaceEnd) {
        // Beyond the address space: one inaccessible block up to 2^64, so a scan
        // that steps by base + size wraps to 0 and stops (as rtld's does).
        base = kAddressSpaceEnd;
        size = 0 - kAddressSpaceEnd;
        type = kMemInaccessible;
    } else if (const Region* r = g_mem.Find(va)) {
        base = r->base;
        size = r->size;
        type = r->type;
        perm = r->perm; // Perm uses Horizon's bit values (R=1, W=2, X=4)
    } else {
        // Unmapped run around va: from the end of the previous region to the next one.
        size = ~0ULL;
        for (const auto& r : g_mem.Regions()) {
            if (r.base + r.size <= va) {
                base = r.base + r.size;
            } else if (r.base > va) {
                size = r.base - base;
                break;
            }
        }
        if (size == ~0ULL) {
            size = kAddressSpaceEnd - base;
        }
    }
    const u32 info[4] = {type, 0, perm, 0};
    g_mem.Write(out, &base, 8);
    g_mem.Write(out + 8, &size, 8);
    g_mem.Write(out + 16, info, sizeof info);
    const u64 zero = 0;
    g_mem.Write(out + 32, &zero, 8);
    c->x[0] = kResultSuccess;
    c->x[1] = 0; // PageInfo
    if (g_trace_svc) {
        Log("svc QueryMemory(0x%llx) -> [0x%llx, +0x%llx) type 0x%x perm %u",
            (unsigned long long)va, (unsigned long long)base, (unsigned long long)size, type, perm);
    }
}

/// Before a fatal Break: the 4-byte buffer nn::diag::detail::Abort passes is the
/// failing Result; the guest frame-pointer chain shows who aborted.
void ReportBreak(GuestContext* c) {
    const auto& x = c->x;
    if (x[2] == 4 && g_mem.Find(x[1])) {
        u32 result = 0;
        g_mem.Read(x[1], &result, 4);
        Log("break: result 0x%x (%04u-%04u)", result, 2000 + (result & 0x1FF), (result >> 9) & 0x1FFF);
    }
    Log("break: lr %s", Describe(x[30]).c_str());
    int depth = 0;
    for (u64 ret : diag::GuestCallChain(x[29], 32)) Log("break:   #%d %s", depth++, Describe(ret).c_str());
}

bool GetInfo(u64 id0, u64 id1, u64& out) {
    switch (id0) {
    case 0: out = 0xF; return true;                   // CoreMask
    case 1: out = 0xFFFFFFFFFFFFFFFFULL; return true; // PriorityMask
    case 2: out = kAliasRegion; return true;
    case 3: out = kAliasRegionSize; return true;
    case 4: out = kHeapRegion; return true;
    case 5: out = kHeapRegionSize; return true;
    case 6: out = kTotalMemory; return true;                         // TotalMemorySize
    case 7: out = UsedMemory() + kSystemResourceSize; return true;   // UsedMemorySize
    case 21: out = kTotalMemory - kSystemResourceSize; return true;  // TotalNonSystemMemorySize
    case 22: out = UsedMemory(); return true;                        // UsedNonSystemMemorySize
    case 8: out = 0; return true;                     // DebuggerAttached
    case 9: out = 0; return true;                     // ResourceLimit (none)
    case 10: out = 0; return true;                    // IdleTickCount
    case 20: out = ProcessLocalRegion(); return true; // UserExceptionContextAddress
    case 24: out = 0; return true;                    // FreeThreadCount (0 without a resource limit, as drippu)
    case 11: {                                        // RandomEntropy
        static std::mt19937_64 rng(0x5EED);
        out = id1 < 4 ? rng() : 0;
        return id1 < 4;
    }
    case 12: out = kAslrRegion; return true;          // AslrRegionAddress
    case 13: out = kAslrRegionSize; return true;
    case 14: out = kStackRegion; return true;
    case 15: out = kStackRegionSize; return true;
    case 16: out = kSystemResourceSize; return true;  // SystemResourceSizeTotal (enables VAMM)
    case 17: out = 0; return true;                    // SystemResourceSizeUsed (page tables: not modelled)
    case 18: out = kProgramId; return true;
    case 23: out = 1; return true;                    // IsApplication
    default: return false;
    }
}



} // namespace
} // namespace rt

using namespace rt;

#define trace(...) RT_TRACE_LINE(__VA_ARGS__)

namespace {
void Svc(GuestContext* c, uint32_t imm);
}

extern "C" thread_local uint64_t rt_trace_pc; // rt.cpp (the lift's --trace-pc)

namespace {
// RT_TRACE_PRESENT: each thread's time in SVCs (with the kernel lock's wait), by SVC, for TakeSvcTimes.
struct SvcTimes {
    u64 ns[0x80] = {}, n[0x80] = {};
};
thread_local SvcTimes t_svc_times;
bool SvcTimesOn() {
    static const bool on = std::getenv("RT_TRACE_PRESENT") != nullptr;
    return on;
}

// RT_THREAD_STATS=1: once a second, each guest thread's time outside SVCs ("busy": running guest code, or
// waiting for a host core) and its largest SVCs, the busiest threads first, as "threads:" log lines. A thread
// counts as busy also for its time in SVCs that work rather than wait (MapPhysicalMemory, IPC ...): listed when
// that or its time outside SVCs is at least 20 ms.
struct ThreadSvcStats {
    const KThread* thread = nullptr;
    std::atomic<u64> ns[0x80] = {};
    std::atomic<u64> since_ns{0}; // when the thread was first seen (busy time counts from there)
    std::atomic<u64> in_svc{0};   // the SVC under way: start time << 7 | SVC (0: none)
    u64 last[0x80] = {};       // the reporter's previous totals
    u64 credited = 0, credited_in_svc = 0; // the SVC under way's time the reporter already counted
    std::string name;          // the reporter's (the game's name for the thread, once it has one)
    std::atomic<u64> gil_wait{0}; // time waiting for the kernel lock on entering SVCs (part of ns[])
    u64 last_gil_wait = 0;
    const volatile u64* trace_pc = nullptr; // the thread's rt_trace_pc (RT_PROFILE_GUEST)
};
// The longest hold of the kernel lock by a non-waiting SVC since the reporter last looked, without the time the
// SVC ran with it released (RunWithoutGil: RomFS and save-data I/O, host waits; counted as held before
// 2026-10-24, which made a PoolThread's save look like a 395 ms hold while others waited 2 ms).
struct LongestHold {
    std::mutex mu;
    u64 ns = 0;
    const ThreadSvcStats* by = nullptr;
    u32 svc = 0;
} g_longest_hold;
std::mutex g_thread_stats_mu;
std::vector<ThreadSvcStats*> g_thread_stats; // never freed (exited threads keep theirs)
thread_local ThreadSvcStats* t_thread_stats = nullptr;
u64 NowNs() {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
const char* SvcName(u32 imm);
void ThreadStatsReporter() {
    for (u64 sec = 1;; ++sec) {
        const u64 from = NowNs();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const u64 to = NowNs();
        struct Row {
            double busy_ms;
            std::string text;
        };
        std::vector<Row> rows;
        std::lock_guard lock(g_thread_stats_mu);
        for (ThreadSvcStats* s : g_thread_stats) {
            const u64 cur = s->in_svc.load(std::memory_order_relaxed);
            u64 total = 0, d[0x80];
            for (u32 i = 0; i < 0x80; ++i) {
                const u64 v = s->ns[i].load(std::memory_order_relaxed);
                d[i] = v - s->last[i], s->last[i] = v;
            }
            if (s->credited && cur != s->credited_in_svc) { // that SVC returned: its whole time is in d
                const u32 i = s->credited_in_svc & 0x7F;
                d[i] -= std::min(d[i], s->credited);
                s->credited = 0;
            }
            if (cur) { // credit the time so far of an SVC still under way (a long wait)
                const u64 began = cur >> 7, part = to - std::min(to, std::max(began, from));
                d[cur & 0x7F] += part;
                s->credited = (cur == s->credited_in_svc ? s->credited : 0) + part;
                s->credited_in_svc = cur;
            }
            for (u32 i = 0; i < 0x80; ++i) total += d[i];
            const u64 start = std::max(from, s->since_ns.load(std::memory_order_relaxed));
            if (start >= to) continue;
            const double busy = (static_cast<double>(to - start) - static_cast<double>(total)) / 1e6;
            const u64 gil = s->gil_wait.load(std::memory_order_relaxed), gil_d = gil - s->last_gil_wait;
            s->last_gil_wait = gil;
            // waits and sleeps: SleepThread, WaitSynchronization, ArbitrateLock, WaitProcessWideKeyAtomic,
            // WaitForAddress; and ExitThread (an exited thread stays in it)
            constexpr u32 kWaits[] = {0x0A, 0x0B, 0x18, 0x1A, 0x1C, 0x34, kHostWaitPresent, kHostWaitGpu};
            u64 working = total;
            for (u32 w : kWaits) working -= d[w];
            if (busy < 20 && working < 20'000'000) continue;
            if (s->name.empty()) s->name = diag::OsThreadName(*s->thread);
            u32 order[0x80];
            for (u32 i = 0; i < 0x80; ++i) order[i] = i;
            std::sort(order, order + 0x80, [&](u32 a, u32 b) { return d[a] > d[b]; });
            char b[160];
            std::snprintf(b, sizeof b, "%s (id %llu) busy %.0f ms", s->name.empty() ? "?" : s->name.c_str(),
                          (unsigned long long)s->thread->id, busy);
            std::string text = b;
            for (u32 k = 0; k < 2 && d[order[k]] >= 1000000; ++k) {
                if (const char* n = SvcName(order[k])) std::snprintf(b, sizeof b, "%s %s %.0f ms", k ? "," : "; in", n, d[order[k]] / 1e6);
                else std::snprintf(b, sizeof b, "%s svc 0x%x %.0f ms", k ? "," : "; in", order[k], d[order[k]] / 1e6);
                text += b;
            }
            if (gil_d >= 1000000) {
                std::snprintf(b, sizeof b, "; kernel lock wait %.0f ms", gil_d / 1e6);
                text += b;
            }
            rows.push_back({busy + working / 1e6, std::move(text)});
        }
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.busy_ms > b.busy_ms; });
        std::string line;
        for (size_t i = 0; i < rows.size() && i < 8; ++i) line += (i ? " | " : "") + rows[i].text;
        {
            std::lock_guard hl(g_longest_hold.mu);
            if (g_longest_hold.ns >= 5000000) {
                const char* n = SvcName(g_longest_hold.svc);
                char b[160];
                std::snprintf(b, sizeof b, " || longest kernel-lock hold %.0f ms: %s in %s", g_longest_hold.ns / 1e6,
                              g_longest_hold.by && !g_longest_hold.by->name.empty() ? g_longest_hold.by->name.c_str() : "?",
                              n ? n : "?");
                line += b;
            }
            g_longest_hold.ns = 0;
        }
        Log("threads: t=%llu s (uptime %.1f s): %s", (unsigned long long)sec, Uptime(), line.c_str());
    }
}
bool IsWaitSvc(u32 imm);

// RT_PROFILE_GUEST=<name>[,<name>...]: a sampling profiler of the guest threads whose names contain one of the
// names (GameThread, RHIThread ...). About every 0.5 ms it reads each one's rt_trace_pc (the lift's
// --trace-pc=func: the guest function it is in, or the call site it last returned to) or the SVC it is in; every
// RT_PROFILE_PERIOD seconds (default 10) it logs where the samples fell: guest code, working SVCs, waits, and the
// top 25 guest functions with their hottest sites. Time in host code a guest function calls (memory and FP
// helpers, HLE functions) counts to that function. Implies RT_THREAD_STATS (the per-thread records).
/// Describe(), but a symbol only when it is near (main's functions are mostly unnamed: the nearest export can
/// be megabytes away). A main address in Ghidra: va - 0x7f04000 (handoff).
std::string NearName(u64 va) {
    std::string d = Describe(va);
    const size_t plus = d.rfind("+0x");
    if (plus != std::string::npos && std::strtoull(d.c_str() + plus + 3, nullptr, 16) >= 0x10000) {
        d.resize(d.find(" ("));
    }
    return d;
}

void GuestProfiler(std::vector<std::string> match) {
    const char* period_env = std::getenv("RT_PROFILE_PERIOD");
    const u64 period_ns = (period_env ? std::strtoull(period_env, nullptr, 10) : 10) * 1'000'000'000ULL;
    std::map<const ThreadSvcStats*, int> chosen;   // 1 profiled, 2 not (decided once named)
    std::map<const ThreadSvcStats*, std::map<u64, u64>> pcs;                  // site -> samples
    std::map<const ThreadSvcStats*, std::array<u64, 3>> kinds;               // guest, working SVC, waiting
    std::map<const ThreadSvcStats*, std::map<u32, u64>> svcs;                // SVC -> samples
    u64 next_report = NowNs() + period_ns;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        {
            std::lock_guard lock(g_thread_stats_mu);
            for (const ThreadSvcStats* s : g_thread_stats) {
                int& c = chosen[s];
                if (c == 0) {
                    const std::string name = diag::OsThreadName(*s->thread);
                    if (name.empty()) continue;
                    c = 2;
                    for (const std::string& m : match) {
                        if (name.find(m) != std::string::npos) c = 1;
                    }
                    if (c == 1) Log("profile: sampling %s (id %llu)", name.c_str(), (unsigned long long)s->thread->id);
                }
                if (c != 1 || !s->trace_pc) continue;
                const u64 in = s->in_svc.load(std::memory_order_relaxed);
                if (in) {
                    const u32 imm = in & 0x7F;
                    ++kinds[s][IsWaitSvc(imm) ? 2 : 1];
                    ++svcs[s][imm];
                } else {
                    ++kinds[s][0];
                    const u64 pc = *s->trace_pc;
                    ++pcs[s][pc];
                }
            }
        }
        if (NowNs() < next_report) continue;
        next_report += period_ns;
        for (auto& [s, sites] : pcs) {
            const auto k = kinds[s];
            const u64 total = k[0] + k[1] + k[2];
            if (!total) continue;
            // sites -> functions (rt_funcs is sorted by offset)
            std::map<u64, std::pair<u64, std::vector<std::pair<u64, u64>>>> funcs; // start -> (samples, sites)
            for (const auto& [pc, n] : sites) {
                const RtFunc* end = rt_funcs + rt_func_count;
                const RtFunc* it = std::upper_bound(rt_funcs, end, pc, [](u64 v, const RtFunc& f) { return v < f.offset; });
                const u64 start = it == rt_funcs ? 0 : (it - 1)->offset;
                auto& f = funcs[start];
                f.first += n;
                f.second.push_back({n, pc});
            }
            std::vector<std::pair<u64, u64>> order; // (samples, start)
            for (const auto& [start, f] : funcs) order.push_back({f.first, start});
            std::sort(order.rbegin(), order.rend());
            std::string svc_list;
            std::vector<std::pair<u64, u32>> so;
            for (const auto& [imm, n] : svcs[s]) so.push_back({n, imm});
            std::sort(so.rbegin(), so.rend());
            for (size_t i = 0; i < so.size() && i < 4; ++i) {
                const char* n = SvcName(so[i].second);
                char b[64];
                std::snprintf(b, sizeof b, "%s%s %.1f %%", i ? ", " : " (", n ? n : "?", 100.0 * so[i].first / total);
                svc_list += b;
            }
            if (!svc_list.empty()) svc_list += ")";
            Log("profile: %s, %llu samples: guest code %.1f %%, working SVCs %.1f %%, waiting %.1f %%%s; %zu functions",
                diag::OsThreadName(*s->thread).c_str(), (unsigned long long)total, 100.0 * k[0] / total,
                100.0 * k[1] / total, 100.0 * k[2] / total, svc_list.c_str(), funcs.size());
            for (size_t i = 0; i < order.size() && i < 25; ++i) {
                auto& f = funcs[order[i].second];
                std::sort(f.second.rbegin(), f.second.rend());
                std::string hot;
                for (size_t j = 0; j < f.second.size() && j < 3; ++j) {
                    char b[48];
                    std::snprintf(b, sizeof b, "%s+0x%llx %.0f %%", j ? ", " : "", (unsigned long long)(f.second[j].second - order[i].second),
                                  100.0 * f.second[j].first / f.first);
                    hot += b;
                }
                Log("profile:   %5.1f %%  %s; sites %s", 100.0 * order[i].first / total,
                    NearName(kImageBase + order[i].second).c_str(), hot.c_str());
            }
            sites.clear();
            kinds[s] = {};
            svcs[s].clear();
        }
    }
}

bool ThreadStatsOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_THREAD_STATS");
        const char* prof = std::getenv("RT_PROFILE_GUEST");
        if ((!e || !*e || *e == '0') && !prof) return false;
        std::thread(ThreadStatsReporter).detach();
        if (prof && *prof) {
            std::vector<std::string> match;
            std::string list = prof;
            for (size_t at = 0; at <= list.size();) {
                const size_t end = std::min(list.find(',', at), list.size());
                if (end > at) match.push_back(list.substr(at, end - at));
                at = end + 1;
            }
            std::thread(GuestProfiler, std::move(match)).detach();
        }
        return true;
    }();
    return on;
}
ThreadSvcStats* MyThreadStats() {
    if (!t_thread_stats) {
        const KThread* t = CurrentThread();
        if (!t) return nullptr;
        t_thread_stats = new ThreadSvcStats;
        t_thread_stats->thread = t;
        t_thread_stats->since_ns = NowNs();
        t_thread_stats->trace_pc = &rt_trace_pc;
        std::lock_guard lock(g_thread_stats_mu);
        g_thread_stats.push_back(t_thread_stats);
    }
    return t_thread_stats;
}
const char* SvcName(u32 imm) {
    switch (imm) {
    case 0x0B: return "SleepThread";
    case 0x11: return "SignalEvent";
    case 0x12: return "ClearEvent";
    case 0x17: return "ResetSignal";
    case 0x18: return "WaitSynchronization";
    case 0x1A: return "ArbitrateLock";
    case 0x1B: return "ArbitrateUnlock";
    case 0x1C: return "WaitProcessWideKeyAtomic";
    case 0x1D: return "SignalProcessWideKey";
    case 0x21: return "SendSyncRequest";
    case 0x2C: return "MapPhysicalMemory";
    case 0x2D: return "UnmapPhysicalMemory";
    case 0x34: return "WaitForAddress";
    case 0x35: return "SignalToAddress";
    case kHostWaitPresent: return "host: present wait";
    case kHostWaitGpu: return "host: GPU wait";
    default: return nullptr;
    }
}
} // namespace

namespace {
// RT_TRACE_WAITS=<thread id or part of its nn::os name>[,...]: each wait of those threads (SleepThread,
// WaitSynchronization, ArbitrateLock, WaitProcessWideKeyAtomic, WaitForAddress) as a row of waits.csv in
// the working directory, plus a row per presented frame, to line waits up with captures:
//   wait,<start uptime ms>,<ms>,<thread id>,<SVC>,<what: key / mutex / address / first handle>,
//        <id of the thread that handed it the mutex or signalled its key, 0: none (timeout, event)>,
//        <guest call chain: return addresses, ';'-separated>
//   present,<uptime ms>,,,,<frame>
// and the NVN layer's submissions (nvn/queue.cpp): a "submit" and a "retire" row each.
struct WaitTrace {
    std::mutex mu;
    std::FILE* csv = nullptr;
    std::vector<std::string> match;
};
WaitTrace* Waits() {
    static WaitTrace* w = [] {
        const char* e = std::getenv("RT_TRACE_WAITS");
        if (!e || !*e) return static_cast<WaitTrace*>(nullptr);
        auto* t = new WaitTrace;
        std::string list = e;
        for (size_t at = 0; at <= list.size();) {
            const size_t end = std::min(list.find(',', at), list.size());
            if (end > at) t->match.push_back(list.substr(at, end - at));
            at = end + 1;
        }
        t->csv = std::fopen("waits.csv", "w");
        if (!t->csv) Fatal("cannot write waits.csv");
        std::fputs("kind,t_ms,ms,thread,svc,what,woken_by,chain\n", t->csv);
        return t;
    }();
    return w;
}
bool IsWaitSvc(u32 imm) {
    return imm == 0x0B || imm == 0x18 || imm == 0x1A || imm == 0x1C || imm == 0x34 || imm == kHostWaitPresent ||
           imm == kHostWaitGpu;
}
/// Whether the calling thread is traced: decided once it has a name (or by its id at once).
bool TracedThread(WaitTrace& w) {
    thread_local int state = 0; // 0 unknown, 1 yes, 2 no
    if (state) return state == 1;
    const KThread* t = CurrentThread();
    if (!t) return false;
    const std::string name = diag::OsThreadName(*t), id = std::to_string(t->id);
    for (const std::string& m : w.match) {
        if (m == id || (!name.empty() && name.find(m) != std::string::npos)) return (state = 1) == 1;
    }
    if (!name.empty()) state = 2;
    return false;
}
} // namespace

void rt::TraceWaitsPresent(u64 frame) {
    WaitTrace* w = Waits();
    if (!w) return;
    std::lock_guard lock(w->mu);
    std::fprintf(w->csv, "present,%.3f,,,,%llu\n", Uptime() * 1e3, (unsigned long long)frame);
    std::fflush(w->csv); // the run usually ends by being killed
}

bool rt::TraceWaitsOn() { return Waits() != nullptr; }

void rt::TraceWaitsLine(const std::string& line) {
    WaitTrace* w = Waits();
    if (!w) return;
    std::lock_guard lock(w->mu);
    std::fprintf(w->csv, "%s\n", line.c_str());
    std::fflush(w->csv);
}

std::string rt::TakeSvcTimes() {
    SvcTimes& t = t_svc_times;
    u32 order[0x80];
    for (u32 i = 0; i < 0x80; ++i) order[i] = i;
    std::sort(order, order + 0x80, [&](u32 a, u32 b) { return t.ns[a] > t.ns[b]; });
    std::string out;
    for (u32 k = 0; k < 3 && t.ns[order[k]]; ++k) {
        const u32 i = order[k];
        char b[96];
        if (const char* name = SvcName(i)) std::snprintf(b, sizeof b, "%s%s %llux %.1f ms", out.empty() ? "" : ", ", name, (unsigned long long)t.n[i], t.ns[i] / 1e6);
        else std::snprintf(b, sizeof b, "%ssvc 0x%x %llux %.1f ms", out.empty() ? "" : ", ", i, (unsigned long long)t.n[i], t.ns[i] / 1e6);
        out += b;
    }
    u64 total = 0;
    for (u32 i = 0; i < 0x80; ++i) total += t.ns[i];
    t = {};
    char b[48];
    std::snprintf(b, sizeof b, "%.1f ms in SVCs", total / 1e6);
    return out.empty() ? b : std::string(b) + ": " + out;
}

namespace rt {
std::vector<std::string> ActiveThreadsSince(u64 since_ns) {
    std::vector<std::string> lines;
    if (!ThreadStatsOn()) return lines;
    const u64 now = NowNs();
    u32 others = 0;
    std::lock_guard lock(g_thread_stats_mu);
    for (const ThreadSvcStats* s : g_thread_stats) {
        const KThread* t = s->thread;
        if (t->state == KThread::State::kExited) continue;
        const u64 in = s->in_svc.load(std::memory_order_relaxed);
        const u64 began = in >> 7;
        if (in && began < since_ns) {
            ++others;
            continue;
        }
        const std::string name = diag::OsThreadName(*t);
        char b[200];
        std::string line;
        if (!in) {
            std::snprintf(b, sizeof b, "%s (id %llu) running guest code in %s", name.empty() ? "?" : name.c_str(),
                          (unsigned long long)t->id, s->trace_pc ? NearName(*s->trace_pc).c_str() : "?");
            line = b;
        } else {
            const u32 imm = in & 0x7F;
            const char* n = SvcName(imm);
            std::snprintf(b, sizeof b, "%s (id %llu) in %s for %.0f ms", name.empty() ? "?" : name.c_str(),
                          (unsigned long long)t->id, n ? n : "an SVC", (now - std::min(now, began)) / 1e6);
            line = b;
            if (t->cv_key || t->mutex_addr || t->arb_addr) {
                std::snprintf(b, sizeof b, " [key 0x%llx mutex 0x%llx address 0x%llx]", (unsigned long long)t->cv_key,
                              (unsigned long long)t->mutex_addr, (unsigned long long)t->arb_addr);
                line += b;
            }
            // the first frames are usually nn::os's wait wrappers: deep enough for the game's own, names cut short
            const auto shortname = [](u64 va) {
                std::string n = NearName(va);
                return n.size() > 60 ? n.substr(0, 57) + "..." : n;
            };
            line += " from " + shortname(t->ctx.x[30]);
            for (u64 ret : diag::GuestCallChain(t->ctx.x[29], 5)) line += " <- " + shortname(ret);
        }
        lines.push_back(std::move(line));
    }
    char b[96];
    std::snprintf(b, sizeof b, "%u other threads waiting since before the last frame", others);
    lines.push_back(b);
    return lines;
}

void RunHostWait(HostWaitKind kind, const std::function<void()>& f) {
    ThreadSvcStats* stats = ThreadStatsOn() ? MyThreadStats() : nullptr;
    if (!stats || stats->in_svc.load(std::memory_order_relaxed)) {
        RunWithoutGil(f);
        return;
    }
    const u64 t0 = NowNs();
    stats->in_svc.store(t0 << 7 | kind, std::memory_order_relaxed);
    RunWithoutGil(f);
    stats->ns[kind].fetch_add(NowNs() - t0, std::memory_order_relaxed);
    stats->in_svc.store(0, std::memory_order_relaxed);
}
} // namespace rt

// SVCs run under the kernel lock (threads.h). ExitThread leaves through
// an exception (ThreadExit, threads.cpp) with the lock held; the thread's exit path expects that.
extern "C" void rt_svc(GuestContext* c, uint32_t imm) {
    const bool timed = (SvcTimesOn() || ThreadStatsOn()) && imm < 0x80;
    const u64 t0 = timed ? NowNs() : 0;
    ThreadSvcStats* stats = timed && ThreadStatsOn() ? MyThreadStats() : nullptr;
    if (stats) stats->in_svc.store(t0 << 7 | imm, std::memory_order_relaxed);
    WaitTrace* waits = IsWaitSvc(imm) ? Waits() : nullptr;
    if (waits && !TracedThread(*waits)) waits = nullptr;
    const double w0 = waits ? Uptime() : 0;
    // SleepThread: ns; WaitSynchronization: handle array; ArbitrateLock: mutex; condvar: key; WaitForAddress: address
    const u64 what = !waits ? 0 : imm == 0x0B || imm == 0x34 ? c->x[0] : c->x[1];
    std::vector<u64> chain = waits ? diag::GuestCallChain(c->x[29], 8) : std::vector<u64>{};
    if (waits) chain.insert(chain.begin(), c->x[30]);
    EnterKernel();
    const u64 t_entered = stats ? NowNs() : 0;
    const u64 unlocked0 = stats ? UnlockedNs() : 0; // RunWithoutGil inside the SVC: not a hold
    if (stats) stats->gil_wait.fetch_add(t_entered - t0, std::memory_order_relaxed);
    if (waits) CurrentThread()->woken_by = 0;
    Svc(c, imm);
    if (stats && !IsWaitSvc(imm) && imm != 0x0A) {
        const u64 held = NowNs() - t_entered - (UnlockedNs() - unlocked0);
        std::lock_guard hl(g_longest_hold.mu);
        if (held > g_longest_hold.ns) g_longest_hold.ns = held, g_longest_hold.by = stats, g_longest_hold.svc = imm;
    }
    if (waits) {
        KThread* me = CurrentThread();
        char b[64];
        std::string desc;
        if (imm == 0x18) { // `what` is the handle array: name the first handle
            u32 h = 0;
            if (what) std::memcpy(&h, g_mem.Raw(what), 4);
            desc = g_handles.Describe(h);
        } else {
            std::snprintf(b, sizeof b, "0x%llx", (unsigned long long)what);
            desc = b;
        }
        std::erase(desc, ',');
        std::string ch;
        for (u64 a : chain) {
            std::snprintf(b, sizeof b, "%s%llx", ch.empty() ? "" : ";", (unsigned long long)a);
            ch += b;
        }
        std::lock_guard lock(waits->mu);
        std::fprintf(waits->csv, "wait,%.3f,%.3f,%llu,%s,%s,%llu,%s\n", w0 * 1e3, (Uptime() - w0) * 1e3,
                     (unsigned long long)me->id, SvcName(imm), desc.c_str(), (unsigned long long)me->woken_by,
                     ch.c_str());
    }
    LeaveKernel();
    if (timed) {
        const u64 ns = NowNs() - t0;
        if (SvcTimesOn()) t_svc_times.ns[imm] += ns, ++t_svc_times.n[imm];
        if (stats) {
            stats->ns[imm].fetch_add(ns, std::memory_order_relaxed);
            stats->in_svc.store(0, std::memory_order_relaxed);
        }
    }
}

namespace {
void Svc(GuestContext* c, uint32_t imm) {
    auto& x = c->x;
    switch (imm) {
    case 0x01: { // SetHeapSize(size) -> addr
        const u64 size = x[1];
        if (size > kHeapRegionSize) {
            Fatal("SetHeapSize(0x%llx) exceeds the heap region", (unsigned long long)size);
        }
        if (g_heap_size == 0 && size) {
            g_mem.Map(kHeapRegion, size, kRead | kWrite, kMemHeap, "heap");
        } else {
            g_mem.Resize("heap", size);
        }
        g_heap_size = size;
        x[0] = kResultSuccess;
        x[1] = kHeapRegion;
        trace("svc SetHeapSize(0x%llx) -> 0x%llx", (unsigned long long)size, (unsigned long long)kHeapRegion);
        break;
    }
    case 0x02: { // SetMemoryPermission(addr, size, perm)
        const bool ok = g_mem.Protect(x[0], x[1], static_cast<u32>(x[2]));
        trace("svc SetMemoryPermission(0x%llx, 0x%llx, %llu) -> %s", (unsigned long long)x[0],
              (unsigned long long)x[1], (unsigned long long)x[2], ok ? "ok" : "invalid range");
        x[0] = ok ? kResultSuccess : kInvalidCurrentMemory;
        break;
    }
    case 0x03: // SetMemoryAttribute(addr, size, mask, value): attributes are not modelled
        trace("svc SetMemoryAttribute(0x%llx, 0x%llx, mask 0x%llx, value 0x%llx) -> ok (ignored)",
              (unsigned long long)x[0], (unsigned long long)x[1], (unsigned long long)x[2],
              (unsigned long long)x[3]);
        x[0] = kResultSuccess;
        break;
    case 0x04:   // MapMemory(dst, src, size): nn::os maps thread stacks into the stack region
    case 0x05: { // UnmapMemory(dst, src, size)
        // Horizon makes src inaccessible while dst aliases it, so moving the
        // contents across (and back on unmap) is equivalent to sharing pages.
        const u64 dst = x[0], src = x[1], size = x[2];
        const Region* s = g_mem.Find(src);
        const bool map = imm == 0x04;
        bool ok;
        if (map) {
            ok = size && (size & 0xFFF) == 0 && s && s->perm == (kRead | kWrite) && src + size <= s->base + s->size &&
                 !g_mem.Find(dst) && !g_mem.Find(dst + size - 1);
            if (ok) {
                g_mem.Map(dst, size, kRead | kWrite, kMemStack, "mapped memory");
                std::memcpy(g_mem.Raw(dst), g_mem.Raw(src), size);
                g_mem.Protect(src, size, 0);
            }
        } else {
            const Region* d = g_mem.Find(dst);
            ok = d && d->base == dst && d->size == size && s && s->perm == 0;
            if (ok) {
                std::memcpy(g_mem.Raw(src), g_mem.Raw(dst), size);
                g_mem.Unmap(dst);
                g_mem.Protect(src, size, kRead | kWrite);
            }
        }
        trace("svc %s(0x%llx, 0x%llx, 0x%llx) -> %s", map ? "MapMemory" : "UnmapMemory", (unsigned long long)dst,
              (unsigned long long)src, (unsigned long long)size, ok ? "ok" : "invalid");
        x[0] = ok ? kResultSuccess : kInvalidCurrentMemory;
        break;
    }
    case 0x06: QueryMemory(c); break;
    case 0x2C:   // MapPhysicalMemory(addr, size): back the unmapped parts of a range in the alias region
    case 0x2D: { // UnmapPhysicalMemory(addr, size)
        const u64 addr = x[0], size = x[1];
        const bool map = imm == 0x2C;
        const bool ok = size && ((addr | size) & 0xFFF) == 0 && addr >= kAliasRegion &&
                        addr + size <= kAliasRegion + kAliasRegionSize;
        if (ok && map) {
            // Map each gap between existing regions; fresh host pages read as zero.
            u64 at = addr;
            while (at < addr + size) {
                if (const Region* r = g_mem.Find(at)) {
                    at = r->base + r->size;
                    continue;
                }
                const u64 end = std::min(addr + size, g_mem.NextRegionBase(at));
                g_mem.MapMerged(at, end - at, kRead | kWrite, kMemNormal, "physical memory");
                // RT_ZERO_CHECK=1 (tests): fresh pages read as zero (UnmapRange's zeroing; any change to it)
                static const bool check = [] {
                    const char* e = std::getenv("RT_ZERO_CHECK");
                    return e && *e && *e != '0';
                }();
                if (check) {
                    const u8* p = g_mem.Raw(at);
                    for (u64 i = 0; i < end - at; i += 8) {
                        u64 v;
                        std::memcpy(&v, p + i, 8);
                        if (v) Fatal("RT_ZERO_CHECK: MapPhysicalMemory at 0x%llx reads 0x%llx at +0x%llx", (unsigned long long)at,
                                     (unsigned long long)v, (unsigned long long)i);
                    }
                }
                g_physical_mapped += end - at;
                at = end;
            }
        } else if (ok) {
            g_mem.ForEachOverlapping(addr, addr + size, [&](const Region& r) {
                const u64 lo = std::max(addr, r.base), hi = std::min(addr + size, r.base + r.size);
                if (r.type == kMemNormal) {
                    g_physical_mapped -= hi - lo;
                }
            });
            g_mem.UnmapRange(addr, size);
        }
        trace("svc %s(0x%llx, 0x%llx) -> %s", map ? "MapPhysicalMemory" : "UnmapPhysicalMemory",
              (unsigned long long)addr, (unsigned long long)size, ok ? "ok" : "invalid");
        x[0] = ok ? kResultSuccess : kInvalidCurrentMemory;
        break;
    }
    case 0x07: trace("svc ExitProcess"); Exit(0);
    case 0x11: case 0x12: case 0x17: { // SignalEvent, ClearEvent, ResetSignal
        auto ev = g_handles.GetAs<KEvent>(static_cast<u32>(x[0]));
        trace("svc %s(%s)", imm == 0x11 ? "SignalEvent" : imm == 0x12 ? "ClearEvent" : "ResetSignal",
              g_handles.Describe(x[0]).c_str());
        if (!ev) {
            x[0] = kInvalidHandle;
            break;
        }
        // ResetSignal of an event that is not signalled fails, as on Horizon (KReadableEvent::Reset):
        // nn::os's TryWait on an auto-clear event is ResetSignal and reads success as "was signalled".
        // Always succeeding made every TryWait report an event (the audio device notification: a
        // device re-query per audio frame, 46 a second).
        const bool fail = imm == 0x17 && !ev->signaled;
        ev->signaled = imm == 0x11;
        x[0] = fail ? kInvalidState : kResultSuccess;
        break;
    }
    case 0x13: { // MapSharedMemory(handle, addr, size, perm)
        auto shm = g_handles.GetAs<KSharedMemory>(static_cast<u32>(x[0]));
        trace("svc MapSharedMemory(%s, 0x%llx, 0x%llx, %llu)", g_handles.Describe(x[0]).c_str(),
              (unsigned long long)x[1], (unsigned long long)x[2], (unsigned long long)x[3]);
        if (!shm) {
            x[0] = kInvalidHandle;
            break;
        }
        g_mem.Map(x[1], x[2], static_cast<u32>(x[3]) & (kRead | kWrite), kMemShared, "shmem:" + shm->name);
        std::memcpy(g_mem.Raw(x[1]), shm->data.data(), std::min<u64>(x[2], shm->data.size()));
        shm->mappings.push_back(x[1]);
        x[0] = kResultSuccess;
        break;
    }
    case 0x14: { // UnmapSharedMemory(handle, addr, size)
        auto shm = g_handles.GetAs<KSharedMemory>(static_cast<u32>(x[0]));
        trace("svc UnmapSharedMemory(%s, 0x%llx)", g_handles.Describe(x[0]).c_str(), (unsigned long long)x[1]);
        if (shm) {
            std::erase(shm->mappings, x[1]);
            g_mem.Unmap(x[1]);
        }
        x[0] = shm ? kResultSuccess : kInvalidHandle;
        break;
    }
    case 0x15: { // CreateTransferMemory(addr, size, perm) -> handle
        auto t = std::make_shared<KTransferMemory>();
        t->address = x[1];
        t->size = x[2];
        t->name = "0x" + std::to_string(x[1]);
        const u32 h = g_handles.Add(t);
        trace("svc CreateTransferMemory(0x%llx, 0x%llx, %llu) -> 0x%x", (unsigned long long)x[1],
              (unsigned long long)x[2], (unsigned long long)x[3], h);
        x[0] = kResultSuccess;
        x[1] = h;
        break;
    }
    case 0x16:
        trace("svc CloseHandle(0x%llx %s)", (unsigned long long)x[0], g_handles.Describe(x[0]).c_str());
        g_handles.Close(static_cast<u32>(x[0]));
        x[0] = kResultSuccess;
        break;
    case 0x1E: x[0] = c->host_mem->read_cntpct(nullptr); break; // GetSystemTick
    case 0x1F: { // ConnectToNamedPort(name) -> session handle
        const std::string name = g_mem.ReadString(x[1], 12);
        auto service = MakeService(name);
        if (!service) {
            Fatal("ConnectToNamedPort(\"%s\"): no HLE implementation", name.c_str());
        }
        const u32 h = OpenSession(service);
        trace("svc ConnectToNamedPort(\"%s\") -> 0x%x", name.c_str(), h);
        x[0] = kResultSuccess;
        x[1] = h;
        break;
    }
    case 0x21: x[0] = rt::SendSyncRequest(c, static_cast<u32>(x[0])); break;
    case 0x24: x[0] = kResultSuccess; x[1] = kProcessId; trace("svc GetProcessId -> 0x%llx", (unsigned long long)x[1]); break;
    case 0x26: // Break(reason, addr, size)
        if (x[0] & 0x80000000ULL) { // notification only
            trace("svc Break(0x%llx) notification", (unsigned long long)x[0]);
            x[0] = kResultSuccess;
            break;
        }
        ReportBreak(c);
        Fatal("svc Break(reason 0x%llx, 0x%llx, 0x%llx)", (unsigned long long)x[0],
              (unsigned long long)x[1], (unsigned long long)x[2]);
    case 0x27: { // OutputDebugString(ptr, size)
        std::string s(x[1], '\0');
        g_mem.Read(x[0], s.data(), x[1]);
        Log("debug: %s", s.c_str());
        x[0] = kResultSuccess;
        break;
    }
    case 0x29: { // GetInfo(id0, handle, id1) -> value
        u64 out = 0;
        const bool ok = GetInfo(x[1], x[3], out);
        trace("svc GetInfo(%llu, %s, %llu) -> %s0x%llx", (unsigned long long)x[1],
              g_handles.Describe(x[2]).c_str(), (unsigned long long)x[3], ok ? "" : "unsupported ",
              (unsigned long long)out);
        x[0] = ok ? kResultSuccess : kInvalidEnumValue;
        x[1] = out;
        break;
    }
    case 0x45: { // CreateEvent() -> (writable, readable); one object serves both ends here
        auto ev = std::make_shared<KEvent>();
        ev->name = "user";
        const u32 h = g_handles.Add(ev);
        x[0] = kResultSuccess;
        x[1] = h;
        x[2] = h;
        trace("svc CreateEvent -> 0x%x", h);
        break;
    }
    default:
        if (ThreadSvc(c, imm)) {
            break;
        }
        Fatal("unimplemented svc 0x%x (x0 0x%llx, x1 0x%llx, x2 0x%llx, x3 0x%llx)", imm,
              (unsigned long long)x[0], (unsigned long long)x[1], (unsigned long long)x[2],
              (unsigned long long)x[3]);
    }
}
} // namespace

#undef trace
