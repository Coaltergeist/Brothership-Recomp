// Guest threads and Horizon's synchronization SVCs.
//
// Every guest thread runs on its own host thread. By default guest code runs in
// parallel, as on the console's cores, and g_gil is a *kernel lock*: it is held
// only while the runtime handles an SVC (and by host threads that touch kernel
// state: GPU fence callbacks, the vsync timer), so the runtime's kernel and
// service code stays single-threaded. What guest code reaches without an SVC is
// thread-safe on its own: memory (Memory's reader/writer lock), the exclusive
// monitor (compare-and-swap), the GPU hooks.
//
// RT_GIL=1 restores the old model for comparison: the lock is held whenever
// guest code runs and is given up only where Horizon could block or switch,
// plus preemption points (indirect branches, system-counter reads, guest loads)
// so a guest spin-wait cannot starve the thread it waits for.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kernel.h"

namespace rt {

/// x30 of a new thread: returning from the entry function ends the thread.
constexpr u64 kThreadExitReturn = 0xFFFF'FFFF'FFFF'E000;

/// The global lock: a mutex that knows whether anyone is waiting for it, so a
/// yielding thread can hand it over (std::mutex alone lets the yielder take it
/// straight back before a woken thread is scheduled).
class GlobalLock {
public:
    void lock() {
        contenders_.fetch_add(1, std::memory_order_relaxed);
        mutex_.lock();
        contenders_.fetch_sub(1, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_relaxed);
    }
    void unlock() { mutex_.unlock(); }
    bool try_lock() {
        if (!mutex_.try_lock()) return false;
        generation_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    int Contenders() const { return contenders_.load(std::memory_order_relaxed); }
    u64 Generation() const { return generation_.load(std::memory_order_relaxed); }

private:
    std::mutex mutex_;
    std::atomic<int> contenders_{0};
    std::atomic<u64> generation_{0};
};

extern GlobalLock g_gil;
using GilLock = std::unique_lock<GlobalLock>;

/// True unless RT_GIL=1: guest code runs without the global lock.
extern const bool g_parallel;

/// Whether this host thread holds the global (kernel) lock.
bool HoldsGil();

/// Around an SVC: in parallel mode, take the kernel lock (and give it back).
void EnterKernel();
void LeaveKernel();
/// RT_TRACE_PRESENT=1: the calling thread's time in SVCs since its last call (kernel-lock waits included),
/// with the three largest SVCs, e.g. "12.3 ms in SVCs: SleepThread 1x 10.1 ms, ..." (svc.cpp).
std::string TakeSvcTimes();
/// RT_TRACE_WAITS: a "present" row in waits.csv for presented frame `frame` (svc.cpp).
void TraceWaitsPresent(u64 frame);
/// RT_TRACE_WAITS: whether waits.csv is written, and a row of it (`line` without the newline; svc.cpp).
bool TraceWaitsOn();
void TraceWaitsLine(const std::string& line);

/// The thread this host thread runs, or null before startup.
KThread* CurrentThread();
GuestContext* CurrentContext();

/// Creates the main thread (handle kMainThreadHandle, TLS at kTlsBase) and
/// makes the calling host thread run it, holding `lock` on g_gil.
std::shared_ptr<KThread> StartMainThread(GilLock& lock);

/// Wakes guest threads blocked in a wait so they re-check their condition.
/// Callable from any host thread, with or without the global lock.
void WakeWaiters();

/// The process local region (GetInfo UserExceptionContextAddress): one TLS-sized
/// slot, as Horizon allocates it. nn::os::SetDyingMessageRegion writes to +0x1C0.
u64 ProcessLocalRegion();

/// Runs f with the global lock released, if this thread holds it: for host-side
/// waits on threads that may need the lock themselves (video_core's GPU thread).
void RunWithoutGil(const std::function<void()>& f);
/// The calling thread's time with the lock released by RunWithoutGil so far (ns), so that a kernel-lock hold
/// measured around an SVC can leave it out (RT_THREAD_STATS' "longest kernel-lock hold").
u64 UnlockedNs();
/// RT_THREAD_STATS' records of the guest threads active since `since_ns` (steady clock): running guest code (the
/// function, from rt_trace_pc), or in an SVC or host wait that began after then (which, for how long, a short
/// guest call chain), one line each; and a last line counting the others (waiting since before). For a snapshot
/// of a long frame (diag.cpp, RT_GAP_SNAPSHOT); empty without RT_THREAD_STATS.
std::vector<std::string> ActiveThreadsSince(u64 since_ns);

/// A host-side wait inside an HLE function, as RunWithoutGil, that RT_THREAD_STATS and RT_PROFILE_GUEST count
/// as a wait (a pseudo-SVC number, named in their reports) instead of busy time. Inside an SVC the SVC keeps it.
enum HostWaitKind : u32 {
    kHostWaitPresent = 0x7E, // the NVN layer: until a window texture is off the screen, the host swapchain
    kHostWaitGpu = 0x7F,     // the NVN layer: until a queue's GPU work is done
};
void RunHostWait(HostWaitKind kind, const std::function<void()>& f);

/// Lets a thread waiting for the global lock run now (if there is one).
void PreemptNow();

/// Called from rt_jump: occasionally lets another thread run.
void MaybePreempt();

/// Ends the current thread (from svcExitThread or a return to kThreadExitReturn): a guest thread unwinds
/// to ThreadMain by an exception (destructors run on every host); the main thread ends the process.
[[noreturn]] void ExitCurrentThread();

/// Every guest thread created so far, for diagnostics (diag.cpp). Takes the
/// kernel lock if it can within half a second, else copies without it.
std::vector<std::shared_ptr<KThread>> ThreadsSnapshot();

/// Handles the thread and synchronization SVCs; false if `imm` is not one.
bool ThreadSvc(GuestContext* c, u32 imm);

} // namespace rt
