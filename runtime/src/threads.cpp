// Guest threads and Horizon's synchronization SVCs (see threads.h for the
// locking model). The arbitration semantics follow the kernel's: mutex words
// hold the owner's handle plus kHasWaiters, condition-variable keys are set to
// 1 while a thread waits on them, and address arbiters compare-and-modify
// 32-bit words.
#include "threads.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "ipc.h"
#include "platform/platform.h"

namespace rt {

GlobalLock g_gil;
const bool g_parallel = std::getenv("RT_GIL") == nullptr;

namespace {

using Clock = std::chrono::steady_clock;

constexpr u32 kHasWaiters = 0x4000'0000;
constexpr u32 kInvalidHandle = 0xE401;   // kernel module 1, description 114
constexpr u32 kTimedOut = 0xEA01;        // 117
constexpr u32 kCancelled = 0xEC01;       // 118
constexpr u32 kInvalidPriority = 0xE001; // 112
constexpr u32 kInvalidState = 0xFA01;    // 125
constexpr u32 kInvalidEnumValue = 0xF001;
constexpr u64 kTlsSize = 0x200;

std::condition_variable_any g_kernel_cv;
std::vector<std::shared_ptr<KThread>> g_threads; // every thread ever created
std::vector<KThread*> g_waiters;                 // threads waiting on a mutex, condvar or address
u64 g_wait_seq = 0, g_next_thread_id = 0x49, g_next_tls = kTlsBase + kTlsSize;
int g_started_threads = 1;

thread_local KThread* t_current = nullptr;
thread_local GilLock* t_lock = nullptr;
/// Thrown by ExitCurrentThread and caught only in ThreadMain: the thread's host stack (generated code, SVC
/// and HLE frames) unwinds the same way on every host, running the destructors on the way. (It was a
/// longjmp: POSIX skips the destructors, Windows runs them.) The generated code has unwind tables
/// (.eh_frame on Linux, .pdata on Windows), which is all the throw needs from it.
struct ThreadExit {};
thread_local bool t_guest_thread = false; // a thread started by ThreadMain (not the main thread)

#define trace(...) RT_TRACE_LINE(__VA_ARGS__)

// RT_WAIT_STATS=1: SleepThread and timed waits per second — how many, the time asked for and the time they
// took — to compare hosts (a host sleep that rounds up to whole milliseconds slows a guest that sleeps briefly).
struct WaitStat {
    std::atomic<u64> n{0}, asked_ns{0}, took_ns{0}, short_n{0}; // short: asked for less than 1 ms
};
WaitStat g_sleep_stat, g_timed_stat;
std::atomic<u64> g_yields{0};
bool WaitStatsOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_WAIT_STATS");
        if (!e || !*e || *e == '0') return false;
        std::thread([] {
            u64 last[2][4] = {}, last_y = 0;
            for (u64 sec = 1;; ++sec) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                WaitStat* st[2] = {&g_sleep_stat, &g_timed_stat};
                u64 d[2][4];
                for (int i = 0; i < 2; ++i) {
                    const u64 v[4] = {st[i]->n.load(), st[i]->asked_ns.load(), st[i]->took_ns.load(), st[i]->short_n.load()};
                    for (int k = 0; k < 4; ++k) d[i][k] = v[k] - last[i][k], last[i][k] = v[k];
                }
                const u64 y = g_yields.load(), dy = y - last_y;
                last_y = y;
                Log("waits: t=%llu s: sleep %llu (%llu < 1 ms; asked %.1f ms, took %.1f ms); timed wait %llu (%llu < 1 ms; "
                    "asked %.1f ms, took %.1f ms); yield %llu",
                    (unsigned long long)sec, (unsigned long long)d[0][0], (unsigned long long)d[0][3], d[0][1] / 1e6,
                    d[0][2] / 1e6, (unsigned long long)d[1][0], (unsigned long long)d[1][3], d[1][1] / 1e6, d[1][2] / 1e6,
                    (unsigned long long)dy);
            }
        }).detach();
        return true;
    }();
    return on;
}
u64 SteadyNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count());
}
void CountWait(WaitStat& st, s64 asked_ns, u64 t0) {
    st.n.fetch_add(1, std::memory_order_relaxed);
    st.asked_ns.fetch_add(static_cast<u64>(asked_ns), std::memory_order_relaxed);
    st.took_ns.fetch_add(SteadyNs() - t0, std::memory_order_relaxed);
    if (asked_ns < 1000000) st.short_n.fetch_add(1, std::memory_order_relaxed);
}

// Guest synchronization words are shared with guest code running in parallel,
// which changes them with LDAXR/STLXR (compare-and-swap here), so the kernel
// reads, writes and updates them atomically too.
u32* Word(u64 va, u32 perm) { return reinterpret_cast<u32*>(g_mem.Access(va, 4, perm)); }
u32 Read32(u64 va) { return __atomic_load_n(Word(va, kRead), __ATOMIC_ACQUIRE); }
void Write32(u64 va, u32 v) { __atomic_store_n(Word(va, kWrite), v, __ATOMIC_RELEASE); }
/// Replaces `expected` with `desired` if the word still holds it; else loads it into `expected`.
bool Cas32(u64 va, u32& expected, u32 desired) {
    return __atomic_compare_exchange_n(Word(va, kRead | kWrite), &expected, desired, false, __ATOMIC_ACQ_REL,
                                       __ATOMIC_ACQUIRE);
}

void Notify() { g_kernel_cv.notify_all(); }

/// Releases the global lock until pred() holds or the timeout (ns; < 0 for none)
/// passes; returns pred(). Waiters re-check at least every 10 ms, so state that
/// HLE code changes without notifying (an event a service signals) is still seen.
template <typename Pred>
bool Wait(s64 timeout_ns, Pred pred) {
    const auto deadline = timeout_ns < 0 ? Clock::time_point::max()
                                         : Clock::now() + std::chrono::nanoseconds(timeout_ns);
    while (!pred()) {
        const auto now = Clock::now();
        if (now >= deadline) {
            return false;
        }
        const auto step = now + std::chrono::milliseconds(10);
        g_kernel_cv.wait_until(*t_lock, std::min(deadline, step));
    }
    return true;
}

/// Lets a thread waiting for the global lock run (if there is one) before continuing.
void Yield() {
    if (g_parallel) { // SleepThread(0): give the host core away, not just the kernel lock
        if (WaitStatsOn()) g_yields.fetch_add(1, std::memory_order_relaxed);
        t_lock->unlock();
        std::this_thread::yield();
        t_lock->lock();
        return;
    }
    if (g_gil.Contenders() == 0) {
        return;
    }
    const u64 generation = g_gil.Generation();
    t_lock->unlock();
    while (g_gil.Generation() == generation && g_gil.Contenders() > 0) {
        std::this_thread::yield();
    }
    t_lock->lock();
}

/// Wakes `t` if it sleeps in Block(): only that thread wakes. (Before, every
/// wake woke every waiting thread, and each took the global lock just to find
/// it had nothing to do; on the title screen 20-40 of 47 threads were queued
/// on the lock at any moment.)
void Poke(KThread* t) {
    { std::lock_guard l(t->wait_m); } // a waiter between its check and its sleep sees the flag
    t->wait_cv.notify_one();
}

/// Block()'s wait: on the thread's own condition variable, with the global lock
/// released; the lock is taken again only once pred() holds or time is up.
template <typename Pred>
bool WaitTargeted(KThread* t, s64 timeout_ns, Pred pred) {
    if (pred()) return true;
    t_lock->unlock();
    {
        std::unique_lock l(t->wait_m);
        if (timeout_ns < 0) {
            t->wait_cv.wait(l, pred);
        } else {
            const u64 t0 = WaitStatsOn() ? SteadyNs() : 0;
            t->wait_cv.wait_for(l, std::chrono::nanoseconds(timeout_ns), pred);
            if (t0) CountWait(g_timed_stat, timeout_ns, t0);
        }
    }
    t_lock->lock();
    return pred();
}

/// Waits until another thread sets `wake` (or a timeout / cancellation).
/// Returns the result to report: wake_result, kTimedOut or kCancelled.
u32 Block(KThread* t, s64 timeout_ns, bool cancellable) {
    t->waiting = true;
    t->wake = false;
    t->wait_seq = ++g_wait_seq;
    g_waiters.push_back(t);
    const bool woke = WaitTargeted(t, timeout_ns, [&] { return t->wake || (cancellable && t->cancel); });
    std::erase(g_waiters, t);
    t->waiting = false;
    if (woke && t->wake) {
        return t->wake_result;
    }
    if (cancellable && t->cancel) {
        t->cancel = false;
        return kCancelled;
    }
    return kTimedOut;
}

void Wake(KThread* t, u32 result) {
    t->wake_result = result;
    t->wake = true;
    Poke(t);
}

/// The highest-priority waiter matching `pred` (lowest number; FIFO among equals).
template <typename Pred>
KThread* BestWaiter(Pred pred) {
    KThread* best = nullptr;
    for (KThread* t : g_waiters) {
        if (!t->wake && pred(t) &&
            (!best || t->priority < best->priority ||
             (t->priority == best->priority && t->wait_seq < best->wait_seq))) {
            best = t;
        }
    }
    return best;
}

bool IsMutexWaiter(const KThread* t, u64 addr) { return t->mutex_addr == addr && t->cv_key == 0; }

/// Hands the mutex word at `addr` to its best waiter, or clears it.
void ReleaseMutex(u64 addr) {
    KThread* next = BestWaiter([&](KThread* t) { return IsMutexWaiter(t, addr); });
    if (!next) {
        Write32(addr, 0);
        return;
    }
    const bool more = std::any_of(g_waiters.begin(), g_waiters.end(), [&](KThread* t) {
        return t != next && !t->wake && IsMutexWaiter(t, addr);
    });
    Write32(addr, next->mutex_tag | (more ? kHasWaiters : 0));
    next->mutex_addr = 0;
    next->woken_by = t_current ? t_current->id : 0;
    Wake(next, kResultSuccess);
}

std::shared_ptr<KThread> ThreadFor(u32 handle) {
    if (handle == kPseudoCurrentThread) {
        for (auto& t : g_threads) {
            if (t.get() == t_current) {
                return t;
            }
        }
    }
    return g_handles.GetAs<KThread>(handle);
}

u64 AllocateTls() {
    const u64 tls = g_next_tls;
    if ((tls & 0xFFF) == 0) {
        g_mem.Map(tls, 0x1000, kRead | kWrite, kMemThreadLocal, "TLS");
    }
    g_next_tls += kTlsSize;
    std::memset(g_mem.Raw(tls), 0, kTlsSize);
    return tls;
}

void ThreadMain(std::shared_ptr<KThread> t) {
    GilLock lock(g_gil);
    t_lock = &lock;
    t_current = t.get();
    // Not named after the guest thread: on Linux a new thread inherits its
    // creator's name, and video_core and audio_core start their helper threads
    // from guest threads (inside SVCs). Thread dumps give each guest thread's host id.
    t->host_tid = platform::CurrentThreadId();
    t_guest_thread = true;
    try {
        if (g_parallel) lock.unlock(); // guest code runs without the kernel lock
        for (;;) {
            rt_jump(&t->ctx, t->ctx.pc);
        }
    } catch (const ThreadExit&) {
    }
    // ExitThread arrives from an SVC (lock held) or a return to kThreadExitReturn (not held).
    if (!lock.owns_lock()) lock.lock();
    t->state = KThread::State::kExited;
    --g_started_threads;
    trace("thread %s (id 0x%llx) exited", t->name.c_str(), (unsigned long long)t->id);
    Notify();
}

bool Signaled(const std::shared_ptr<KObject>& obj) {
    if (auto ev = std::dynamic_pointer_cast<KEvent>(obj)) {
        return ev->signaled;
    }
    if (auto th = std::dynamic_pointer_cast<KThread>(obj)) {
        return th->state == KThread::State::kExited;
    }
    return false;
}

void WaitSynchronization(GuestContext* c) {
    auto& x = c->x;
    const u64 handles_va = x[1], count = x[2];
    const s64 timeout = static_cast<s64>(x[3]);
    std::vector<u32> handles(count);
    if (count) {
        g_mem.Read(handles_va, handles.data(), count * 4);
    }
    std::vector<std::shared_ptr<KObject>> objs;
    std::string names;
    for (u32 h : handles) {
        names += (names.empty() ? "" : ", ") + g_handles.Describe(h);
        auto obj = h == kPseudoCurrentThread ? std::shared_ptr<KObject>(ThreadFor(h)) : g_handles.Get(h);
        if (!obj) {
            x[0] = kInvalidHandle;
            trace("svc WaitSynchronization([%s]) -> invalid handle", names.c_str());
            return;
        }
        objs.push_back(std::move(obj));
    }
    KThread* self = t_current;
    auto signaled_index = [&]() -> s64 {
        for (size_t i = 0; i < objs.size(); ++i) {
            if (Signaled(objs[i])) {
                return static_cast<s64>(i);
            }
        }
        return -1;
    };
    if (self->cancel) {
        self->cancel = false;
        x[0] = kCancelled;
        trace("svc WaitSynchronization([%s]) -> cancelled", names.c_str());
        return;
    }
    s64 index = signaled_index();
    if (index < 0 && timeout != 0) {
        trace("svc WaitSynchronization([%s], %lld ns) waits", names.c_str(), (long long)timeout);
        self->waiting = true;
        Wait(timeout, [&] { return self->cancel || (index = signaled_index()) >= 0; });
        self->waiting = false;
    }
    if (index >= 0) {
        x[0] = kResultSuccess;
        x[1] = static_cast<u64>(index);
        trace("svc WaitSynchronization([%s]) -> index %lld", names.c_str(), (long long)index);
    } else if (self->cancel) {
        self->cancel = false;
        x[0] = kCancelled;
        trace("svc WaitSynchronization([%s]) -> cancelled", names.c_str());
    } else {
        x[0] = kTimedOut;
        trace("svc WaitSynchronization([%s]) -> timed out", names.c_str());
    }
}

} // namespace

KThread* CurrentThread() { return t_current; }
bool HoldsGil() { return t_lock && t_lock->owns_lock(); }
void EnterKernel() {
    if (g_parallel && t_lock && !t_lock->owns_lock()) t_lock->lock();
}
void LeaveKernel() {
    if (g_parallel && t_lock && t_lock->owns_lock()) t_lock->unlock();
}
void WakeWaiters() { Notify(); }
GuestContext* CurrentContext() { return t_current ? &t_current->ctx : nullptr; }

std::shared_ptr<KThread> StartMainThread(GilLock& lock) {
    auto t = std::make_shared<KThread>();
    t->name = "main";
    t->id = 0x48;
    t->tls = kTlsBase;
    t->state = KThread::State::kStarted;
    g_handles.AddAt(kMainThreadHandle, t);
    g_threads.push_back(t);
    t_lock = &lock;
    t_current = t.get();
    t->host_tid = platform::CurrentThreadId();
    return t;
}

std::vector<std::shared_ptr<KThread>> ThreadsSnapshot() {
    // The holder of the kernel lock may be the thread that hangs: wait a little,
    // then read without it (a dump is worth the small risk).
    for (int i = 0; i < 50; ++i) {
        if (g_gil.try_lock()) {
            auto copy = g_threads;
            g_gil.unlock();
            return copy;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Log("(the kernel lock is held: reading the thread list without it)");
    return g_threads;
}

u64 ProcessLocalRegion() {
    static const u64 plr = AllocateTls();
    return plr;
}

thread_local u64 t_unlocked_ns = 0;

u64 UnlockedNs() { return t_unlocked_ns; }

void RunWithoutGil(const std::function<void()>& f) {
    if (!t_lock || !t_lock->owns_lock()) {
        f();
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    t_lock->unlock();
    f();
    t_lock->lock();
    t_unlocked_ns += static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

void PreemptNow() {
    if (!g_parallel && g_started_threads > 1) {
        Yield();
    }
}

void MaybePreempt() {
    if (g_parallel) return; // threads run in parallel: nothing to hand over
    thread_local u32 calls = 0;
    if (++calls >= 4096) {
        calls = 0;
        if (g_started_threads > 1) {
            Yield();
        }
    }
}

void ExitCurrentThread() {
    if (!t_guest_thread) { // the main thread
        trace("svc ExitThread (main)");
        Exit(0);
    }
    throw ThreadExit{};
}

bool ThreadSvc(GuestContext* c, u32 imm) {
    auto& x = c->x;
    KThread* self = t_current;
    switch (imm) {
    case 0x08: { // CreateThread(entry, arg, stack_top, priority, core) -> handle
        const s32 priority = static_cast<s32>(x[4]), core = static_cast<s32>(x[5]);
        if (priority < 0 || priority > 63) {
            x[0] = kInvalidPriority;
            break;
        }
        auto t = std::make_shared<KThread>();
        t->id = g_next_thread_id++;
        t->priority = priority;
        t->core = core < 0 ? 0 : core; // -2: the process's default core
        t->tls = AllocateTls();
        t->ctx.host_mem = c->host_mem;
        t->ctx.pending_svc = ~0ULL;
        t->ctx.x[0] = x[2];
        t->ctx.x[30] = kThreadExitReturn;
        t->ctx.x[31] = x[3];
        t->ctx.pc = x[1];
        t->ctx.tpidrro_el0 = t->tls;
        char name[32];
        std::snprintf(name, sizeof name, "0x%llx", (unsigned long long)t->id);
        t->name = name;
        g_threads.push_back(t);
        const u32 h = g_handles.Add(t);
        trace("svc CreateThread(entry %s, arg 0x%llx, sp 0x%llx, prio %d, core %d) -> 0x%x (id 0x%llx)",
              Describe(x[1]).c_str(), (unsigned long long)x[2], (unsigned long long)x[3], priority, core, h,
              (unsigned long long)t->id);
        x[0] = kResultSuccess;
        x[1] = h;
        break;
    }
    case 0x09: { // StartThread(handle)
        auto t = ThreadFor(static_cast<u32>(x[0]));
        trace("svc StartThread(%s)", g_handles.Describe(x[0]).c_str());
        if (!t || t->state != KThread::State::kCreated) {
            x[0] = t ? kInvalidState : kInvalidHandle;
            break;
        }
        t->state = KThread::State::kStarted;
        ++g_started_threads;
        std::thread(ThreadMain, t).detach();
        x[0] = kResultSuccess;
        break;
    }
    case 0x0A: // ExitThread
        trace("svc ExitThread (%s)", self->name.c_str());
        ExitCurrentThread();
    case 0x0B: { // SleepThread(ns); 0, -1, -2 yield
        const s64 ns = static_cast<s64>(x[0]);
        if (ns <= 0) {
            Yield();
        } else {
            // A plain sleep: no wake-up can end it, so it must not join the waiters.
            t_lock->unlock();
            const u64 t0 = WaitStatsOn() ? SteadyNs() : 0;
            platform::SleepFor(std::chrono::nanoseconds(ns));
            if (t0) CountWait(g_sleep_stat, ns, t0);
            t_lock->lock();
        }
        break;
    }
    case 0x0C: { // GetThreadPriority(handle) -> priority
        auto t = ThreadFor(static_cast<u32>(x[1]));
        x[0] = t ? kResultSuccess : kInvalidHandle;
        if (t) {
            x[1] = static_cast<u64>(t->priority);
        }
        trace("svc GetThreadPriority -> %lld", t ? (long long)t->priority : -1LL);
        break;
    }
    case 0x0D: { // SetThreadPriority(handle, priority)
        auto t = ThreadFor(static_cast<u32>(x[0]));
        trace("svc SetThreadPriority(%s, %d)", g_handles.Describe(x[0]).c_str(), static_cast<s32>(x[1]));
        if (t) {
            t->priority = static_cast<s32>(x[1]);
        }
        x[0] = t ? kResultSuccess : kInvalidHandle;
        break;
    }
    case 0x0E: { // GetThreadCoreMask(handle) -> (core, mask)
        auto t = ThreadFor(static_cast<u32>(x[2]));
        x[0] = t ? kResultSuccess : kInvalidHandle;
        if (t) {
            x[1] = static_cast<u64>(t->core);
            x[2] = 1ULL << t->core;
        }
        break;
    }
    case 0x0F: { // SetThreadCoreMask(handle, core, mask): affinity is not modelled
        auto t = ThreadFor(static_cast<u32>(x[0]));
        trace("svc SetThreadCoreMask(%s, %d, 0x%llx)", g_handles.Describe(x[0]).c_str(), static_cast<s32>(x[1]),
              (unsigned long long)x[2]);
        if (t && static_cast<s32>(x[1]) >= 0) {
            t->core = static_cast<s32>(x[1]);
        }
        x[0] = t ? kResultSuccess : kInvalidHandle;
        break;
    }
    case 0x10: // GetCurrentProcessorNumber
        x[0] = static_cast<u64>(self->core);
        break;
    case 0x18:
        WaitSynchronization(c);
        break;
    case 0x19: { // CancelSynchronization(thread handle)
        auto t = ThreadFor(static_cast<u32>(x[0]));
        trace("svc CancelSynchronization(%s)", g_handles.Describe(x[0]).c_str());
        if (t) {
            t->cancel = true;
            Poke(t.get()); // in Block()
            Notify(); // in WaitSynchronization
        }
        x[0] = t ? kResultSuccess : kInvalidHandle;
        break;
    }
    case 0x1A: { // ArbitrateLock(owner handle, address, tag)
        const u32 owner = static_cast<u32>(x[0]), tag = static_cast<u32>(x[2]);
        const u64 addr = x[1];
        if (Read32(addr) != (owner | kHasWaiters)) {
            x[0] = kResultSuccess; // the word changed: the caller retries
            break;
        }
        self->mutex_addr = addr;
        self->mutex_tag = tag;
        self->cv_key = 0;
        x[0] = Block(self, -1, false);
        break;
    }
    case 0x1B: // ArbitrateUnlock(address)
        ReleaseMutex(x[0]);
        x[0] = kResultSuccess;
        break;
    case 0x1C: { // WaitProcessWideKeyAtomic(mutex address, key, tag, timeout)
        const u64 mutex = x[0], key = x[1];
        const u32 tag = static_cast<u32>(x[2]);
        const s64 timeout = static_cast<s64>(x[3]);
        // Horizon's order: mark the condition variable as waited on *before*
        // releasing the mutex. nn::os's Signal/Broadcast skip the SVC when the key
        // reads 0, so with the mutex released first, a thread running in parallel
        // could take it, change the state, see 0 and not signal: a lost wake-up
        // (it stalled the frame threads on the title screen). Not counted as a
        // mutex waiter (cv_key != 0), so ReleaseMutex does not hand it the lock.
        self->mutex_addr = mutex;
        self->mutex_tag = tag;
        self->cv_key = key;
        Write32(key, 1);
        ReleaseMutex(mutex);
        u32 r = Block(self, timeout, false);
        if (r == kTimedOut && self->cv_key == 0 && self->mutex_addr != 0) {
            // Signalled just before the timeout and now queued for the mutex: wait for it.
            r = Block(self, -1, false);
        }
        if (r == kTimedOut) {
            self->cv_key = 0;
            self->mutex_addr = 0; // libnx/SDK relock the mutex themselves after a timeout
        }
        x[0] = r;
        break;
    }
    case 0x1D: { // SignalProcessWideKey(key, count)
        const u64 key = x[0];
        const s32 count = static_cast<s32>(x[1]);
        for (s32 n = 0; count <= 0 || n < count; ++n) {
            KThread* t = BestWaiter([&](KThread* w) { return w->cv_key == key; });
            if (!t) {
                break;
            }
            t->cv_key = 0;
            t->woken_by = self->id;
            // Take the mutex for t if it is free, else mark it contended; guest code
            // may lock or unlock it concurrently, so as one atomic update.
            u32 word = Read32(t->mutex_addr);
            for (;;) {
                const u32 desired = word == 0 ? t->mutex_tag : (word | kHasWaiters);
                if (Cas32(t->mutex_addr, word, desired)) break;
            }
            if (word == 0) {
                t->mutex_addr = 0;
                Wake(t, kResultSuccess);
            } // else: t now waits for the mutex
        }
        if (!BestWaiter([&](KThread* w) { return w->cv_key == key; })) {
            Write32(key, 0);
        }
        x[0] = kResultSuccess;
        break;
    }
    case 0x25: { // GetThreadId(handle) -> id
        auto t = ThreadFor(static_cast<u32>(x[1]));
        x[0] = t ? kResultSuccess : kInvalidHandle;
        if (t) {
            x[1] = t->id;
        }
        break;
    }
    case 0x32: // SetThreadActivity(handle, paused): pausing is not modelled
        trace("svc SetThreadActivity(%s, %llu) -> ok (ignored)", g_handles.Describe(x[0]).c_str(),
              (unsigned long long)x[1]);
        x[0] = kResultSuccess;
        break;
    case 0x34: { // WaitForAddress(address, type, value, timeout)
        const u64 addr = x[0];
        const u32 type = static_cast<u32>(x[1]);
        const s32 value = static_cast<s32>(x[2]);
        const s64 timeout = static_cast<s64>(x[3]);
        const s32 cur = static_cast<s32>(Read32(addr));
        bool ok;
        switch (type) {
        case 0: ok = cur < value; break; // WaitIfLessThan
        case 1: {                        // DecrementAndWaitIfLessThan
            u32 seen = static_cast<u32>(cur);
            for (;;) {
                ok = static_cast<s32>(seen) < value;
                if (!ok || Cas32(addr, seen, seen - 1)) break;
            }
            break;
        }
        case 2: ok = cur == value; break; // WaitIfEqual
        default: x[0] = kInvalidEnumValue; return true;
        }
        if (!ok) {
            x[0] = kInvalidState;
        } else if (timeout == 0) {
            x[0] = kTimedOut;
        } else {
            self->arb_addr = addr;
            x[0] = Block(self, timeout, true);
            self->arb_addr = 0;
        }
        break;
    }
    case 0x35: { // SignalToAddress(address, type, value, count)
        const u64 addr = x[0];
        const u32 type = static_cast<u32>(x[1]);
        const s32 value = static_cast<s32>(x[2]), count = static_cast<s32>(x[3]);
        auto is_waiter = [&](KThread* w) { return w->arb_addr == addr; };
        if (type == 1 || type == 2) {
            u32 expected = static_cast<u32>(value);
            s32 new_value = value + 1; // SignalAndIncrementIfEqual, or no waiters
            if (type == 2) {           // SignalAndModifyByWaitingCountIfEqual (as the kernel computes it)
                const s32 waiters = static_cast<s32>(std::count_if(
                    g_waiters.begin(), g_waiters.end(), [&](KThread* w) { return !w->wake && is_waiter(w); }));
                if (waiters > 0) {
                    new_value = count <= 0 ? value - 2 : (waiters - 1 < count ? value - 1 : value);
                }
            }
            if (!Cas32(addr, expected, static_cast<u32>(new_value))) { // compare and modify in one step
                x[0] = kInvalidState;
                break;
            }
        } else if (type != 0) {
            x[0] = kInvalidEnumValue;
            break;
        }
        for (s32 n = 0; count <= 0 || n < count; ++n) {
            KThread* t = BestWaiter(is_waiter);
            if (!t) {
                break;
            }
            t->arb_addr = 0;
            Wake(t, kResultSuccess);
        }
        x[0] = kResultSuccess;
        break;
    }
    default:
        return false;
    }
    return true;
}

} // namespace rt
