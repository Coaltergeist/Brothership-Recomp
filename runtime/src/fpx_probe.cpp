// FPX probe counters (fpx_probe.h; only called by a build with -DRT_FPX_PROBE=ON). Per guest thread: how often
// drippu's FP helpers took the host fast path, and why not when they did not: FPSR.IXC still clear (a cumulative
// flag could change), FPCR not default (rounding mode, FZ, DN, AH ... : RECOMP_FPX_FPCR_MASK), or open but the
// result was one the fast path does not keep (tiny, huge, NaN, inf). Logged every 10 s for the threads with
// counts, with the FPCR values seen.
#include "fpx_probe.h"

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "diag.h"
#include "kernel.h"
#include "rt.h"
#include "threads.h"

namespace rt {
namespace {

constexpr u64 kFpcrMask = 0x07C8FF07; // drippu's RECOMP_FPX_FPCR_MASK without its inhibit bit
enum { kHit, kClosedIxc, kClosedFpcr, kNotKept, kKinds };
constexpr const char* kKindNames[kKinds] = {"fast path", "closed: FPSR.IXC clear", "closed: FPCR", "open, result not kept"};

struct Counts {
    std::atomic<u64> n[kKinds]{};       // single writer (the thread), read by the reporter
    std::atomic<u64> last_fpcr{0};
    std::atomic<u64> fpcr_changes{0};
    const KThread* thread = nullptr;
    u64 reported[kKinds] = {};
};

std::mutex g_mu;
std::vector<Counts*> g_all; // never freed
thread_local Counts* t_counts = nullptr;

void Bump(std::atomic<u64>& a) { a.store(a.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed); }

void Reporter() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::lock_guard lock(g_mu);
        for (Counts* c : g_all) {
            u64 d[kKinds], total = 0;
            for (int i = 0; i < kKinds; ++i) {
                const u64 now = c->n[i].load(std::memory_order_relaxed);
                d[i] = now - c->reported[i];
                c->reported[i] = now;
                total += d[i];
            }
            if (total < 1000) continue;
            std::string line;
            for (int i = 0; i < kKinds; ++i) {
                char b[80];
                std::snprintf(b, sizeof b, "%s%s %.1f %%", i ? ", " : "", kKindNames[i], 100.0 * d[i] / total);
                line += b;
            }
            const std::string name = c->thread ? diag::OsThreadName(*c->thread) : "?";
            Log("fpx probe: %s: %.0f per s; %s; FPCR now 0x%llx (%llu changes so far)", name.empty() ? "?" : name.c_str(),
                total / 10.0, line.c_str(), (unsigned long long)c->last_fpcr.load(),
                (unsigned long long)c->fpcr_changes.load());
        }
    }
}

Counts& Mine() {
    if (!t_counts) {
        t_counts = new Counts;
        t_counts->thread = CurrentThread();
        std::lock_guard lock(g_mu);
        if (g_all.empty()) {
            Log("fpx probe: counting drippu's FP fast path (RT_FPX_PROBE build)");
            std::thread(Reporter).detach();
        }
        g_all.push_back(t_counts);
    }
    return *t_counts;
}

} // namespace
} // namespace rt

extern "C" void rt_fpx_probe(int kind, unsigned long long fpsr, unsigned long long fpcr) {
    using namespace rt;
    Counts& c = Mine();
    const u64 masked = fpcr & kFpcrMask;
    if (masked != c.last_fpcr.load(std::memory_order_relaxed)) {
        c.last_fpcr.store(masked, std::memory_order_relaxed);
        Bump(c.fpcr_changes);
    }
    if (kind == 1) return Bump(c.n[kHit]);
    if (!(fpsr & 16)) return Bump(c.n[kClosedIxc]);
    if (masked) return Bump(c.n[kClosedFpcr]);
    Bump(c.n[kNotKept]);
}
