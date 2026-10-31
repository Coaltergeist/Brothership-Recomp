// Kernel objects and the process handle table.
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rt.h"

namespace rt {

class Service;

struct KObject {
    virtual ~KObject() = default;
    virtual const char* Kind() const = 0;
    std::string name;
};

/// A guest thread: its register context, and the kernel's view of what it is
/// waiting for. Every field is protected by the global lock (threads.h).
struct KThread : KObject {
    const char* Kind() const override { return "thread"; }
    enum class State { kCreated, kStarted, kExited };

    GuestContext ctx{};
    u64 id = 0, tls = 0;
    u64 host_tid = 0; // the host thread running it (platform::CurrentThreadId), for dumps
    s32 priority = 44, core = 0;
    State state = State::kCreated;

    // Waiting (see threads.cpp). A thread waits for `wake`, set by whoever
    // ends the wait together with `wake_result`. A blocked thread sleeps on its
    // own wait_cv with the global lock released, so wake/cancel are atomic and
    // a waker pokes only this thread.
    bool waiting = false;
    std::atomic<bool> wake{false}, cancel{false};
    std::mutex wait_m;
    std::condition_variable wait_cv;
    u32 wake_result = 0;
    u64 mutex_addr = 0; // to be handed this mutex word (ArbitrateLock, or after a condvar signal)
    u32 mutex_tag = 0;  // the value it writes there as owner
    u64 cv_key = 0;     // waiting on this condition variable key
    u64 arb_addr = 0;   // WaitForAddress
    u64 wait_seq = 0;   // order of arrival, for FIFO among equal priorities
    u64 woken_by = 0;   // id of the thread that last handed it a mutex or signalled its key (RT_TRACE_WAITS)
};

struct KEvent : KObject {
    const char* Kind() const override { return "event"; }
    // Atomic so host threads (audio_core, timers) can signal without the
    // global lock; they then call WakeWaiters() (threads.h).
    std::atomic<bool> signaled{false};
};

/// Shared memory owned by an HLE service. The service writes through Write(),
/// which updates every place the guest has mapped it.
struct KSharedMemory : KObject {
    const char* Kind() const override { return "shmem"; }
    explicit KSharedMemory(u64 size) : data(size) {}
    void Write(u64 offset, const void* src, u64 size);
    std::vector<u8> data;
    std::vector<u64> mappings; // guest addresses
};

struct KTransferMemory : KObject {
    const char* Kind() const override { return "tmem"; }
    u64 address = 0, size = 0;
};

/// A domain's object table. Every session cloned from a domain session belongs
/// to the same domain and shares it.
struct KDomain {
    std::map<u32, std::shared_ptr<Service>> objects;
    u32 next_object_id = 1;
};

/// Client end of an IPC session. A domain session multiplexes several service
/// objects by object id; otherwise the session talks to `object` alone.
struct KSession : KObject {
    const char* Kind() const override { return "session"; }
    std::shared_ptr<Service> object;
    std::shared_ptr<KDomain> domain; // null: not a domain
};

class HandleTable {
public:
    u32 Add(std::shared_ptr<KObject> obj);
    void AddAt(u32 handle, std::shared_ptr<KObject> obj) { objects_[handle] = std::move(obj); }
    std::shared_ptr<KObject> Get(u32 handle) const;
    template <typename T>
    std::shared_ptr<T> GetAs(u32 handle) const {
        return std::dynamic_pointer_cast<T>(Get(handle));
    }
    bool Close(u32 handle);
    std::string Describe(u64 handle) const;

private:
    static constexpr u32 kFirstHandle = 0x0003'0001, kHandleSlots = 0x4000 - 3;
    std::map<u32, std::shared_ptr<KObject>> objects_;
    u32 next_ = kFirstHandle;
};

extern HandleTable g_handles;

constexpr u32 kMainThreadHandle = 0x0002'0001;
constexpr u64 kPseudoCurrentThread = 0xFFFF8000, kPseudoCurrentProcess = 0xFFFF8001;

/// Opens a new client session to `object` and returns its handle.
u32 OpenSession(std::shared_ptr<Service> object);

} // namespace rt
