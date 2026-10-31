#include "diag.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "perf.h"
#include "platform/platform.h"
#include "threads.h"
#include "window.h"

namespace rt::diag {

// The first name-like string in the thread's nn::os::ThreadType (pointer at
// TLS+0x1F8), or "".
std::string OsThreadName(const KThread& t) {
    u64 tt = 0;
    if (!g_mem.Find(t.tls + 0x1F8)) return "";
    std::memcpy(&tt, g_mem.Raw(t.tls + 0x1F8), 8);
    if (!tt || !g_mem.Find(tt) || !g_mem.Find(tt + 0x1FF)) return "";
    const u8* p = g_mem.Raw(tt);
    for (int i = 0; i < 0x200; ++i) {
        int len = 0;
        while (i + len < 0x200 && p[i + len] >= 0x20 && p[i + len] < 0x7F) ++len;
        if (len >= 4 && (i + len == 0x200 || p[i + len] == 0)) return std::string(reinterpret_cast<const char*>(p + i), len);
        i += len;
    }
    return "";
}

namespace {

std::atomic<bool> g_dump_requested{false};

void RequestDump() { // from a signal handler: only set the flag
    g_dump_requested.store(true, std::memory_order_relaxed);
}

void DumpGuestThread(const KThread& t) {
    const GuestContext& c = t.ctx;
    const std::string os_name = OsThreadName(t);
    const char* state = t.state == KThread::State::kExited ? "exited" : t.state == KThread::State::kCreated ? "created"
                        : t.waiting                         ? "blocked"
                                                            : "running";
    Log("--- guest thread %s%s%s%s (host tid %llu), %s, lr %s, sp 0x%llx", t.name.c_str(), os_name.empty() ? "" : " \"",
        os_name.c_str(), os_name.empty() ? "" : "\"", (unsigned long long)t.host_tid, state, Describe(c.x[30]).c_str(),
        (unsigned long long)c.x[31]);
    if (t.cv_key || t.mutex_addr || t.arb_addr) {
        Log("    waiting: cv key 0x%llx, mutex 0x%llx, arbiter 0x%llx", (unsigned long long)t.cv_key,
            (unsigned long long)t.mutex_addr, (unsigned long long)t.arb_addr);
        if (const u64 at = t.cv_key ? t.cv_key : t.mutex_addr ? t.mutex_addr : t.arb_addr) {
            char b[512];
            int n = std::snprintf(b, sizeof b, "    memory at 0x%llx-0x40:", (unsigned long long)at);
            for (u64 a = at - 0x40; a < at + 0x40; a += 4) {
                u32 v = 0xDEADDEAD;
                if (g_mem.Find(a)) std::memcpy(&v, g_mem.Raw(a), 4);
                n += std::snprintf(b + n, sizeof b - static_cast<size_t>(n), "%s%08x", a == at ? " [" : " ", v);
            }
            Log("%s", b);
        }
    }
    // The context of a running thread is only as fresh as the last time the
    // generated code stored it: the chain is a hint, exact for blocked threads.
    int depth = 0;
    for (u64 ret : GuestCallChain(c.x[29], 16)) Log("    #%d %s", depth++, Describe(ret).c_str());
}

void DumpAll() {
    Log("thread dump at t=%.1f s, after frame %llu", Uptime(), (unsigned long long)perf::Frames());
    for (const auto& t : ThreadsSnapshot()) {
        if (t->state != KThread::State::kExited) DumpGuestThread(*t);
    }
    Log("host stacks (resolve nxrt frames with addr2line -f -C -e nxrt <offset>):");
    std::fflush(stderr);
    platform::DumpHostStacks();
    Log("end of thread dump");
}

/// RT_GAP_SNAPSHOT=<ms> (default 400 with a window, 0 off; needs RT_THREAD_STATS): when no frame has come for that
/// long, and again at four times it, which guest threads were active since the last frame and in what
/// (ActiveThreadsSince), as "gap:" lines - the new-game run's cutscene frames of 0.5-2.6 s (2026-10-24) had no
/// thread busy all along and the RHIThread in its present wait, so the per-second lines could not say why.
void GapSnapshot(double idle, int nth) {
    const u64 now_ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now().time_since_epoch()).count());
    const u64 since = now_ns - static_cast<u64>(idle * 1e9);
    const std::vector<std::string> lines = ActiveThreadsSince(since);
    if (lines.empty()) return;
    Log("gap: no frame for %.2f s (snapshot %d, after frame %llu, t=%.2f s)", idle, nth,
        (unsigned long long)perf::Frames(), Uptime());
    for (const std::string& l : lines) Log("gap:   %s", l.c_str());
}

void Run() {
    platform::SetCurrentThreadName("rt diagnostics");
    const char* w = std::getenv("RT_WATCHDOG");
    const double limit = w ? std::atof(w) : 30.0;
    const char* g = std::getenv("RT_GAP_SNAPSHOT");
    double gap = -1;    // decided at the first frame: this thread starts before the window opens (rt.cpp)
    bool fired = false; // for the current stall
    int snapshots = 0;  // for the current gap
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (g_dump_requested.exchange(false, std::memory_order_relaxed)) DumpAll();
        if (perf::Frames() == 0) continue;
        if (gap < 0) {
            gap = (g ? std::atof(g) : window::IsOpen() ? 400.0 : 0.0) / 1e3;
            if (ActiveThreadsSince(0).empty()) gap = 0; // no RT_THREAD_STATS
            if (gap > 0) Log("diag: gap snapshots on, at %.0f ms without a frame and at four times it", gap * 1e3);
            else Log("diag: gap snapshots off (they need a window or RT_GAP_SNAPSHOT=<ms>, and RT_THREAD_STATS=1)");
        }
        const double since_frame = Uptime() - perf::LastFrameUptime();
        if (gap > 0) {
            if (since_frame < gap) snapshots = 0;
            else if (snapshots == 0 || (snapshots == 1 && since_frame >= 4 * gap)) GapSnapshot(since_frame, ++snapshots);
        }
        if (limit <= 0) continue;
        const double idle = since_frame;
        if (idle < limit) {
            fired = false;
        } else if (!fired) {
            fired = true;
            Log("watchdog: no frame for %.1f s (after frame %llu, t=%.1f s); dumping threads", idle,
                (unsigned long long)perf::Frames(), Uptime());
            DumpAll();
            std::this_thread::sleep_for(std::chrono::seconds(3));
            DumpAll();
        }
    }
}

} // namespace

void Start() {
    platform::InstallDumpTrigger(RequestDump);
    std::thread(Run).detach();
}

std::vector<u64> GuestCallChain(u64 fp, int max_depth) {
    std::vector<u64> out;
    for (int depth = 0; depth < max_depth && fp && g_mem.Find(fp) && g_mem.Find(fp + 15); ++depth) {
        u64 frame[2]; // saved x29, saved x30
        std::memcpy(frame, g_mem.Raw(fp), sizeof frame);
        out.push_back(frame[1]);
        if (frame[0] <= fp) break; // the chain only grows towards older frames
        fp = frame[0];
    }
    return out;
}

} // namespace rt::diag
