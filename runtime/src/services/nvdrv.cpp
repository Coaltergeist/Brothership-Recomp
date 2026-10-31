// nvdrv: the GPU driver interface (the NVN driver's view of the kernel driver).
//
// The devices follow drippu's (yuzu's) core/hle/service/nvdrv/devices, ported
// onto this runtime's IPC and kernel objects, and drive drippu's video_core
// (runtime/gpu, ../gpu.h): the GPU MMU is a Tegra::MemoryManager per address
// space, nvmap handles are pinned into Host1x's SMMU (device address space),
// syncpoints are Host1x's, and submissions go to Tegra::GPU channels.
//
// Like everywhere else in the runtime, handlers run under the global lock.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <functional>
#include <optional>
#include "../gpu.h"
#include "../ipc.h"
#include "../services.h"
#include "../threads.h"

#include "common/settings.h"
#include "core/core.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/puller.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"

namespace rt {
namespace {

// NvResult (drippu nvdata.h).
constexpr u32 kNvSuccess = 0x0, kNvBadParameter = 0x4, kNvTimeout = 0x5, kNvInsufficientMemory = 0x6,
              kNvBadValue = 0xB, kNvAlreadyAllocated = 0xD, kNvBusy = 0xE, kNvConfigVarNotFound = 0x30006,
              kNvAccessDenied = 0x30010;

constexpr u32 kMaxSyncpoints = 192, kMaxNvEvents = 64;
constexpr u64 kPageSize = 0x1000;

struct NvFence {
    s32 id;
    u32 value;
};

u64 AlignUp(u64 x, u64 a) { return (x + a - 1) & ~(a - 1); }

// ---------------------------------------------------------------------------
// Ioctl argument plumbing (drippu ioctl_serialization.h). An ioctl has a fixed
// argument struct at the start of the input, optionally followed by an array;
// Ioctl2 adds an inline input array and Ioctl3 an inline output array.

struct IoctlBufs {
    std::vector<u8> in, inl_in;   // what the guest sent
    std::vector<u8> out, inl_out; // sized to the guest's out buffers
};

template <typename T>
std::vector<T> ReadArray(const std::vector<u8>& v, size_t from) {
    std::vector<T> a(from < v.size() ? (v.size() - from) / sizeof(T) : 0);
    if (!a.empty()) std::memcpy(a.data(), v.data() + from, a.size() * sizeof(T));
    return a;
}

template <typename T>
void WriteBack(std::vector<u8>& out, size_t at, const T* p, size_t bytes) {
    if (out.size() > at) std::memcpy(out.data() + at, p, std::min(out.size() - at, bytes));
}

/// Fixed argument only.
template <typename T>
u32 Fixed(IoctlBufs& b, const std::function<u32(T&)>& f) {
    T p{};
    std::memcpy(&p, b.in.data(), std::min(sizeof(T), b.in.size()));
    const u32 r = f(p);
    WriteBack(b.out, 0, &p, sizeof(T));
    return r;
}

/// Fixed argument followed by an array of V in the input (and output).
template <typename T, typename V>
u32 FixedVariable(IoctlBufs& b, const std::function<u32(T&, std::vector<V>&)>& f) {
    T p{};
    const size_t off = std::min(sizeof(T), b.in.size());
    std::memcpy(&p, b.in.data(), off);
    auto var = ReadArray<V>(b.in, off);
    const u32 r = f(p, var);
    WriteBack(b.out, 0, &p, sizeof(T));
    WriteBack(b.out, off, var.data(), var.size() * sizeof(V));
    return r;
}

/// Fixed argument, plus an array of V in the inline input (Ioctl2).
template <typename T, typename V>
u32 FixedInlIn(IoctlBufs& b, const std::function<u32(T&, std::vector<V>&)>& f) {
    T p{};
    std::memcpy(&p, b.in.data(), std::min(sizeof(T), b.in.size()));
    auto var = ReadArray<V>(b.inl_in, 0);
    const u32 r = f(p, var);
    WriteBack(b.out, 0, &p, sizeof(T));
    return r;
}

/// Fixed argument, plus an array of V written to the inline output (Ioctl3).
template <typename T, typename V>
u32 FixedInlOut(IoctlBufs& b, const std::function<u32(T&, std::vector<V>&)>& f) {
    T p{};
    std::memcpy(&p, b.in.data(), std::min(sizeof(T), b.in.size()));
    std::vector<V> var(b.inl_out.size() / sizeof(V));
    const u32 r = f(p, var);
    WriteBack(b.out, 0, &p, sizeof(T));
    WriteBack(b.inl_out, 0, var.data(), var.size() * sizeof(V));
    return r;
}

std::shared_ptr<KEvent> NewEvent(const std::string& name) {
    auto e = std::make_shared<KEvent>();
    e->name = name;
    return e;
}

// ---------------------------------------------------------------------------
// video_core access.

/// video_core's System; the GPU comes up on first use.
Core::System& Sys() {
    gpu::Init();
    return gpu::System();
}

/// GPU-side ("host") syncpoints: their values, and actions run when one reaches a threshold.
Tegra::Host1x::SyncpointManager& Host() {
    return Sys().Host1x().GetSyncpointManager();
}

// nvdrv's own syncpoint bookkeeping (drippu nvdrv/core/syncpoint_manager.cpp):
// which ids are reserved, and the min (GPU reached) / max (submitted) values.
class SyncpointManager {
public:
    SyncpointManager() {
        Reserve(26, true); // VBlank0
        Reserve(27, true); // VBlank1
        for (u32 id : {0xCu /*VIC*/, 0x36u /*NvDec*/, 0x37u /*NvJpg*/}) Reserve(id, false);
    }
    u32 Allocate(bool client_managed) {
        for (u32 i = 1; i < kMaxSyncpoints; ++i) {
            if (!sp_[i].reserved) return Reserve(i, client_managed);
        }
        Fatal("nvdrv: out of syncpoints");
    }
    void Free(u32 id) { sp_.at(id).reserved = false; }
    bool IsAllocated(u32 id) const { return id < kMaxSyncpoints && sp_[id].reserved; }
    bool HasExpired(u32 id, u32 threshold) const {
        const auto& s = sp_.at(id);
        if (s.interface_managed) return static_cast<s32>(s.min - threshold) >= 0;
        return (s.max - threshold) >= (s.min - threshold);
    }
    bool IsFenceSignalled(NvFence f) const { return HasExpired(static_cast<u32>(f.id), f.value); }
    u32 IncrementMaxExt(u32 id, u32 amount) { return sp_.at(id).max += amount; }
    u32 ReadMin(u32 id) const { return sp_.at(id).min; }
    u32 UpdateMin(u32 id) { return sp_.at(id).min = Host().GetHostSyncpointValue(id); }
    NvFence Fence(u32 id) const { return {static_cast<s32>(id), sp_.at(id).max}; }

private:
    u32 Reserve(u32 id, bool client_managed) {
        auto& s = sp_.at(id);
        if (s.reserved) Fatal("nvdrv: syncpoint %u reserved twice", id);
        s.reserved = true;
        s.interface_managed = client_managed;
        return id;
    }
    struct Info {
        u32 min = 0, max = 0;
        bool reserved = false, interface_managed = false;
    };
    std::array<Info, kMaxSyncpoints> sp_{};
};

SyncpointManager& Syncpoints() {
    static SyncpointManager s;
    return s;
}

// ---------------------------------------------------------------------------
// nvmap handles (drippu nvdrv/core/nvmap.cpp). Handles and ids are the same
// value, as in drippu (single process).

struct NvMapHandle {
    u64 size = 0, orig_size = 0, aligned_size = 0;
    u64 address = 0; // guest VA of the backing memory
    u32 align = 0, flags = 0, refs = 1;
    u8 kind = 0;
    bool allocated = false;
    DAddr d_address = 0; // SMMU (device) address while pinned
    s32 pins = 0;
};

struct NvMapFile {
    std::map<u32, NvMapHandle> handles;
    u32 next_id = 1;
    NvMapHandle* Get(u32 id) {
        auto it = handles.find(id);
        return it == handles.end() ? nullptr : &it->second;
    }
};

NvMapFile& NvMap() {
    static NvMapFile f;
    return f;
}

constexpr u64 kSmmuBigPage = 0x10000; // drippu nvmap.cpp BIG_PAGE_SIZE

/// The SMMU's id for the guest process (drippu: Container::OpenSession).
Core::Asid ProcessAsid() {
    static const Core::Asid asid = Sys().Host1x().MemoryManager().RegisterProcess(&Sys().ApplicationMemory());
    return asid;
}

/// Maps a handle's memory into the SMMU on first use (drippu NvMap::PinHandle,
/// without the unmap queue: a handle stays mapped until it is freed).
DAddr PinHandle(NvMapHandle& h) {
    if (!h.d_address) {
        // The GPU reaches memory through the device backing (rt.h).
        g_mem.MakeDeviceVisible(h.address, h.aligned_size);
        auto& smmu = Sys().Host1x().MemoryManager();
        h.d_address = smmu.Allocate(AlignUp(h.aligned_size, kSmmuBigPage));
        if (!h.d_address) Fatal("nvmap: SMMU address space exhausted");
        smmu.Map(h.d_address, h.address, h.aligned_size, ProcessAsid(), true);
    }
    ++h.pins;
    return h.d_address;
}

void UnpinHandle(u32 id) {
    if (auto* h = NvMap().Get(id); h && h->pins > 0) --h->pins;
}

void UnmapHandle(NvMapHandle& h) {
    if (!h.d_address) return;
    auto& smmu = Sys().Host1x().MemoryManager();
    smmu.Unmap(h.d_address, h.aligned_size);
    smmu.Free(h.d_address, AlignUp(h.aligned_size, kSmmuBigPage));
    h.d_address = 0;
}

/// Page-granular first-fit allocator (stands in for Common::FlatAllocator).
/// Allocate returns 0 on failure; the range never starts at 0.
class PageAllocator {
public:
    PageAllocator(u32 start, u32 limit) : start_(start), limit_(limit) { free_[start] = limit - start; }
    u32 Start() const { return start_; }
    u32 Limit() const { return limit_; }
    u32 Allocate(u32 n) {
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it->second < n) continue;
            const u32 at = it->first, left = it->second - n;
            free_.erase(it);
            if (left) free_[at + n] = left;
            return at;
        }
        return 0;
    }
    void AllocateFixed(u32 at, u32 n) {
        auto it = free_.upper_bound(at);
        if (it == free_.begin()) Fatal("nvdrv: fixed GPU VA page 0x%x is not free", at);
        --it;
        const u32 b = it->first, len = it->second;
        if (at + n > b + len) Fatal("nvdrv: fixed GPU VA range 0x%x+0x%x is not free", at, n);
        free_.erase(it);
        if (at > b) free_[b] = at - b;
        if (b + len > at + n) free_[at + n] = b + len - (at + n);
    }
    void Free(u32 at, u32 n) {
        auto [it, ok] = free_.emplace(at, n);
        if (!ok) Fatal("nvdrv: GPU VA page 0x%x freed twice", at);
        if (auto next = std::next(it); next != free_.end() && at + n == next->first) {
            it->second += next->second;
            free_.erase(next);
        }
        if (it != free_.begin()) {
            if (auto prev = std::prev(it); prev->first + prev->second == at) {
                prev->second += it->second;
                free_.erase(it);
            }
        }
    }

private:
    u32 start_, limit_;
    std::map<u32, u32> free_; // start page -> page count
};

// ---------------------------------------------------------------------------
// Devices.

class Device {
public:
    explicit Device(std::string path) : path(std::move(path)) {}
    virtual ~Device() = default;
    virtual u32 Ioctl1(u32 gc, IoctlBufs&) { return Unknown(1, gc); }
    virtual u32 Ioctl2(u32 gc, IoctlBufs&) { return Unknown(2, gc); }
    virtual u32 Ioctl3(u32 gc, IoctlBufs&) { return Unknown(3, gc); }
    virtual std::shared_ptr<KEvent> QueryEvent(u32 id) {
        Fatal("nvdrv QueryEvent(%u) on %s not implemented", id, path.c_str());
    }
    const std::string path;

protected:
    [[noreturn]] u32 Unknown(int kind, u32 gc) {
        Fatal("nvdrv Ioctl%d group 0x%02x cmd 0x%02x on %s not implemented", kind, gc >> 8, gc & 0xFF,
              path.c_str());
    }
};

constexpr u32 GC(u32 group, u32 cmd) { return group << 8 | cmd; }

std::map<s32, std::shared_ptr<Device>>& Fds() {
    static std::map<s32, std::shared_ptr<Device>> fds;
    return fds;
}

template <typename T>
std::shared_ptr<T> DeviceAt(s32 fd) {
    auto it = Fds().find(fd);
    return it == Fds().end() ? nullptr : std::dynamic_pointer_cast<T>(it->second);
}

// /dev/nvmap ----------------------------------------------------------------

class NvMapDevice : public Device {
public:
    using Device::Device;
    u32 Ioctl1(u32 gc, IoctlBufs& b) override {
        auto& f = NvMap();
        switch (gc) {
        case GC(1, 0x01): { // Create
            struct P { u32 size, handle; };
            return Fixed<P>(b, [&](P& p) {
                p.handle = f.next_id++;
                auto& h = f.handles[p.handle];
                h.size = AlignUp(p.size, kPageSize);
                h.orig_size = p.size;
                Log("nv   nvmap Create size 0x%x -> handle %u", p.size, p.handle);
                return kNvSuccess;
            });
        }
        case GC(1, 0x03): { // FromId
            struct P { u32 id, handle; };
            return Fixed<P>(b, [&](P& p) {
                auto* h = p.id ? f.Get(p.id) : nullptr;
                if (!h) return kNvBadValue;
                ++h->refs;
                p.handle = p.id;
                return kNvSuccess;
            });
        }
        case GC(1, 0x04): { // Alloc
            struct P { u32 handle, heap_mask, flags, align; u8 kind, pad[7]; u64 address; };
            static_assert(sizeof(P) == 0x20);
            return Fixed<P>(b, [&](P& p) {
                if (!p.handle || ((p.align - 1) & p.align)) return kNvBadValue;
                p.align = std::max<u32>(p.align, kPageSize);
                auto* h = f.Get(p.handle);
                if (!h) return kNvBadValue;
                if (h->allocated) return kNvInsufficientMemory;
                if (!p.address) Fatal("nvmap Alloc without a CPU address (handle %u)", p.handle);
                h->flags = p.flags & ~4u; // keep_uncached_after_free only applies without an address
                h->kind = p.kind;
                h->align = p.align;
                h->aligned_size = AlignUp(h->size, p.align);
                h->address = p.address;
                h->allocated = true;
                Log("nv   nvmap Alloc handle %u at %s size 0x%llx align 0x%x kind %u", p.handle,
                    Describe(p.address).c_str(), (unsigned long long)h->size, p.align, p.kind);
                return kNvSuccess;
            });
        }
        case GC(1, 0x05): { // Free
            struct P { u32 handle, pad; u64 address; u32 size, flags; };
            static_assert(sizeof(P) == 0x18);
            return Fixed<P>(b, [&](P& p) {
                auto* h = p.handle ? f.Get(p.handle) : nullptr;
                if (!h) return kNvSuccess;
                if (--h->refs == 0) {
                    UnmapHandle(*h);
                    p.address = h->address;
                    p.size = static_cast<u32>(h->size);
                    p.flags = h->flags & 1; // map_uncached
                    f.handles.erase(p.handle);
                }
                return kNvSuccess;
            });
        }
        case GC(1, 0x09): { // Param
            struct P { u32 handle, param, result; };
            return Fixed<P>(b, [&](P& p) {
                auto* h = p.handle ? f.Get(p.handle) : nullptr;
                if (!h) return kNvBadValue;
                switch (p.param) {
                case 1: p.result = static_cast<u32>(h->orig_size); break;
                case 2: p.result = h->align; break;
                case 3: p.result = static_cast<u32>(-22); break; // Base: EINVAL
                case 4: p.result = h->allocated ? 0x40000000 : 0; break;
                case 5: p.result = h->kind; break;
                case 6: p.result = 0; break; // IsSharedMemMapped
                default: return kNvBadValue;
                }
                return kNvSuccess;
            });
        }
        case GC(1, 0x0E): { // GetId
            struct P { u32 id, handle; };
            return Fixed<P>(b, [&](P& p) {
                if (!p.handle) return kNvBadValue;
                if (!f.Get(p.handle)) return kNvAccessDenied;
                p.id = p.handle;
                return kNvSuccess;
            });
        }
        }
        return Unknown(1, gc);
    }
};

// /dev/nvhost-ctrl ----------------------------------------------------------

union SyncpointEventValue {
    u32 raw;
    struct { u32 partial_slot : 4, syncpoint_id : 28; };
    struct { u16 slot; u16 syncpoint_id_for_allocation : 12, event_allocated : 1, : 3; };
};
static_assert(sizeof(SyncpointEventValue) == 4);

enum class EventState { kAvailable, kWaiting, kCancelling, kSignalling, kSignalled, kCancelled };

class NvHostCtrl : public Device {
public:
    using Device::Device;
    u32 Ioctl1(u32 gc, IoctlBufs& b) override {
        switch (gc) {
        case GC(0, 0x1B): { // NvOsGetConfigU32
            struct P { char domain[0x41], param[0x41], config[0x101]; };
            return Fixed<P>(b, [&](P& p) {
                p.domain[0x40] = p.param[0x40] = 0;
                Log("nv   ctrl GetConfig %s!%s -> not found", p.domain, p.param);
                return kNvConfigVarNotFound; // production units have no config vars
            });
        }
        case GC(0, 0x1C): { // EventClear
            return Fixed<SyncpointEventValue>(b, [&](SyncpointEventValue& p) { return ClearEventWait(p.slot); });
        }
        case GC(0, 0x1D): case GC(0, 0x1E): { // EventWaitAsync (allocates a slot), EventWait
            struct P { NvFence fence; u32 timeout; SyncpointEventValue value; };
            const bool alloc = gc == GC(0, 0x1D);
            return Fixed<P>(b, [&](P& p) { return EventWait(p.fence, p.timeout, p.value, alloc); });
        }
        case GC(0, 0x1F): { // EventRegister
            return Fixed<u32>(b, [&](u32& id) {
                if (id >= kMaxNvEvents) return kNvBadParameter;
                if (events_[id].kevent) {
                    if (u32 r = FreeEvent(id)) return r;
                }
                CreateEvent(id);
                return kNvSuccess;
            });
        }
        case GC(0, 0x20): { // EventUnregister
            return Fixed<u32>(b, [&](u32& id) { return FreeEvent(id & 0xFF); });
        }
        case GC(0, 0x21): { // EventUnregisterBatch
            return Fixed<u64>(b, [&](u64& mask) {
                for (u64 m = mask; m; m &= m - 1) {
                    if (u32 r = FreeEvent(static_cast<u32>(std::countr_zero(m)))) return r;
                }
                return kNvSuccess;
            });
        }
        }
        return Unknown(1, gc);
    }

    std::shared_ptr<KEvent> QueryEvent(u32 id) override {
        const SyncpointEventValue v{.raw = id};
        const bool allocated = v.event_allocated;
        const u32 slot = allocated ? v.partial_slot : v.slot;
        const u32 syncpt = allocated ? v.syncpoint_id_for_allocation : v.syncpoint_id;
        if (slot < kMaxNvEvents && events_[slot].kevent && events_[slot].syncpt == syncpt) {
            return events_[slot].kevent;
        }
        Fatal("nvdrv ctrl QueryEvent: slot %u syncpoint %u not registered", slot, syncpt);
    }

private:
    struct Event {
        std::shared_ptr<KEvent> kevent;
        EventState status = EventState::kAvailable;
        u32 fails = 0, syncpt = 0, value = 0;
        std::optional<Tegra::Host1x::SyncpointManager::ActionHandle> wait_handle;
        bool Busy() const {
            return status == EventState::kWaiting || status == EventState::kCancelling ||
                   status == EventState::kSignalling;
        }
    };

    u32 EventWait(NvFence& fence, u32 timeout, SyncpointEventValue& value, bool alloc) {
        auto& sm = Syncpoints();
        const u32 id = static_cast<u32>(fence.id);
        const u32 slot_in = value.raw;
        if (id >= kMaxSyncpoints) return kNvBadParameter;
        auto unmark = [&] { if (!alloc && slot_in < kMaxNvEvents) events_[slot_in].fails = 0; };
        if (fence.value == 0) {
            if (sm.IsAllocated(id)) value.raw = sm.ReadMin(id);
            unmark();
            return kNvSuccess;
        }
        if (sm.IsFenceSignalled(fence)) {
            value.raw = sm.ReadMin(id);
            unmark();
            return kNvSuccess;
        }
        if (const u32 v = sm.UpdateMin(id); sm.IsFenceSignalled(fence)) {
            value.raw = v;
            unmark();
            return kNvSuccess;
        }
        const u32 slot = alloc ? FindFreeEvent(id) : slot_in;
        if (alloc) value.raw = 0;
        if (slot >= kMaxNvEvents) return kNvBadParameter;
        auto& ev = events_[slot];
        if (ev.fails > 2) {
            // drippu blocks here until the GPU reaches the value (Host1x WaitHost).
            if (static_cast<s32>(Host().GetHostSyncpointValue(id) - fence.value) < 0) {
                if (!Settings::values.use_asynchronous_gpu_emulation.GetValue()) {
                    // Synchronous GPU: submitted work has run, so it never will.
                    Fatal("nvdrv ctrl EventWait: syncpoint %u is at %u and will not reach %u", id,
                          Host().GetHostSyncpointValue(id), fence.value);
                }
                // Asynchronous GPU: it is still behind. The GPU thread's fence
                // callbacks take the global lock, so wait without it.
                RunWithoutGil([&] { Host().WaitHost(id, fence.value); });
            }
            value.raw = fence.value;
            ev.fails = 0;
            return kNvSuccess;
        }
        if (timeout == 0) return kNvTimeout;
        if (!ev.kevent || ev.Busy()) return kNvBadParameter;

        value.raw = 0;
        ev.status = EventState::kWaiting;
        ev.syncpt = id;
        ev.value = fence.value;
        if (alloc) {
            value.syncpoint_id_for_allocation = id;
            value.event_allocated = 1;
        } else {
            value.syncpoint_id = id;
        }
        value.raw |= slot;
        auto handle = Host().RegisterHostAction(id, fence.value, [this, slot] {
            // On a video_core thread (asynchronous fences) this runs outside the global lock.
            auto signal = [&] {
                auto& e = events_[slot];
                if (e.status == EventState::kWaiting) e.kevent->signaled = true;
                e.status = EventState::kSignalled;
                WakeWaiters(); // without it a waiter sees the fence up to 10 ms late
            };
            if (HoldsGil()) { // e.g. a synchronous submission inside an SVC
                signal();
            } else {          // a video_core thread, or a guest thread outside an SVC
                std::lock_guard lock(g_gil);
                signal();
            }
        });
        if (ev.status == EventState::kWaiting) ev.wait_handle = handle; // else it already ran
        Log("nv   ctrl EventWait syncpoint %u >= %u -> event slot %u", id, fence.value, slot);
        return kNvTimeout;
    }

    u32 ClearEventWait(u32 slot) {
        if (slot >= kMaxNvEvents) return kNvBadParameter;
        auto& ev = events_[slot];
        if (ev.status == EventState::kWaiting && ev.wait_handle) {
            Host().DeregisterHostAction(ev.syncpt, *ev.wait_handle);
            Syncpoints().UpdateMin(ev.syncpt);
        }
        ev.wait_handle.reset();
        ev.fails++;
        ev.status = EventState::kCancelled;
        if (ev.kevent) ev.kevent->signaled = false;
        return kNvSuccess;
    }

    void CreateEvent(u32 slot) {
        auto& ev = events_[slot];
        ev = Event{};
        ev.kevent = NewEvent("nvhost-ctrl event " + std::to_string(slot));
    }

    u32 FreeEvent(u32 slot) {
        if (slot >= kMaxNvEvents) return kNvBadParameter;
        auto& ev = events_[slot];
        if (!ev.kevent) return kNvSuccess;
        if (ev.Busy()) return kNvBusy;
        ev = Event{};
        return kNvSuccess;
    }

    u32 FindFreeEvent(u32 syncpt) {
        u32 slot = kMaxNvEvents, free_slot = kMaxNvEvents;
        for (u32 i = 0; i < kMaxNvEvents; ++i) {
            const auto& ev = events_[i];
            if (ev.kevent) {
                if (!ev.Busy()) {
                    slot = i;
                    if (ev.syncpt == syncpt) return i;
                }
            } else if (free_slot == kMaxNvEvents) {
                free_slot = i;
            }
        }
        if (free_slot < kMaxNvEvents) {
            CreateEvent(free_slot);
            return free_slot;
        }
        if (slot < kMaxNvEvents) return slot;
        Fatal("nvdrv ctrl: no free event slot");
    }

    std::array<Event, kMaxNvEvents> events_{};
};

// /dev/nvhost-ctrl-gpu ------------------------------------------------------

struct GpuCharacteristics {
    u32 arch, impl, rev, num_gpc;
    u64 l2_cache_size, on_board_video_memory_size;
    u32 num_tpc_per_gpc, bus_type, big_page_size, compression_page_size, pde_coverage_bit_count,
        available_big_page_sizes, gpc_mask, sm_arch_sm_version, sm_arch_spa_version,
        sm_arch_warp_count, gpu_va_bit_count, reserved;
    u64 flags;
    u32 twod_class, threed_class, compute_class, gpfifo_class, inline_to_memory_class,
        dma_copy_class, max_fbps_count, fbp_en_mask, max_ltc_per_fbp, max_lts_per_ltc,
        max_tex_per_tpc, max_gpc_count, rop_l2_en_mask_0, rop_l2_en_mask_1;
    u64 chipname, gr_compbit_store_base_hw;
};
static_assert(sizeof(GpuCharacteristics) == 0xA0);

// Tegra X1 (GM20B) values, as drippu reports them.
constexpr GpuCharacteristics kGm20b{
    .arch = 0x120, .impl = 0xB, .rev = 0xA1, .num_gpc = 1,
    .l2_cache_size = 0x40000, .on_board_video_memory_size = 0,
    .num_tpc_per_gpc = 2, .bus_type = 0x20, .big_page_size = 0x20000,
    .compression_page_size = 0x20000, .pde_coverage_bit_count = 0x1B,
    .available_big_page_sizes = 0x30000, .gpc_mask = 1, .sm_arch_sm_version = 0x503,
    .sm_arch_spa_version = 0x503, .sm_arch_warp_count = 0x80, .gpu_va_bit_count = 0x28,
    .reserved = 0, .flags = 0x55,
    .twod_class = 0x902D, .threed_class = 0xB197, .compute_class = 0xB1C0,
    .gpfifo_class = 0xB06F, .inline_to_memory_class = 0xA140, .dma_copy_class = 0xB0B5,
    .max_fbps_count = 1, .fbp_en_mask = 0, .max_ltc_per_fbp = 2, .max_lts_per_ltc = 1,
    .max_tex_per_tpc = 0, .max_gpc_count = 1, .rop_l2_en_mask_0 = 0x21D70,
    .rop_l2_en_mask_1 = 0, .chipname = 0x6230326D67 /* "gm20b" */, .gr_compbit_store_base_hw = 0,
};

class NvHostCtrlGpu : public Device {
public:
    using Device::Device;
    struct Characteristics { u64 buf_size, buf_addr; GpuCharacteristics gc; };
    struct TpcMasks { u32 mask_buf_size, pad0; u64 mask_buf_addr; u32 tpc_mask, pad1; };
    static_assert(sizeof(TpcMasks) == 24);

    u32 Ioctl1(u32 gc, IoctlBufs& b) override {
        constexpr u32 G = 'G';
        switch (gc) {
        case GC(G, 0x01): return Fixed<u32>(b, [](u32& size) { size = 1; return kNvSuccess; }); // ZcullGetCtxSize
        case GC(G, 0x02): { // ZcullGetInfo
            return Fixed<std::array<u32, 10>>(b, [](std::array<u32, 10>& p) {
                p = {0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xC0, 0x20, 0x40, 0x10};
                return kNvSuccess;
            });
        }
        case GC(G, 0x03): { // ZbcSetTable
            struct P { u32 color_ds[4], color_l2[4], depth, format, type; };
            return Fixed<P>(b, [&](P& p) { return ZbcSet(p.color_ds, p.color_l2, p.depth, p.format, p.type); });
        }
        case GC(G, 0x04): { // ZbcQueryTable
            struct P { u32 color_ds[4], color_l2[4], depth, ref_cnt, format, type, index_size; };
            return Fixed<P>(b, [&](P& p) {
                const auto& t = p.type == 1 ? colors_ : p.type == 2 ? depths_ : colors_;
                if (p.type < 1 || p.type > 2 || p.index_size >= t.size()) return kNvBadParameter;
                const auto& e = t[p.index_size];
                std::copy(e.ds.begin(), e.ds.end(), p.color_ds);
                std::copy(e.l2.begin(), e.l2.end(), p.color_l2);
                p.depth = e.depth;
                p.ref_cnt = e.refs;
                p.format = e.format;
                p.index_size = static_cast<u32>(t.size());
                return kNvSuccess;
            });
        }
        case GC(G, 0x05): { // GetCharacteristics
            return Fixed<Characteristics>(b, [](Characteristics& p) {
                p.buf_size = sizeof(GpuCharacteristics);
                p.buf_addr = 0xdeadbeef; // must not be 0; unused
                p.gc = kGm20b;
                return kNvSuccess;
            });
        }
        case GC(G, 0x06): { // GetTpcMasks
            return Fixed<TpcMasks>(b, [](TpcMasks& p) {
                if (p.mask_buf_size) p.tpc_mask = 3;
                return kNvSuccess;
            });
        }
        case GC(G, 0x07): return Fixed<u64>(b, [](u64&) { return kNvSuccess; }); // FlushL2
        case GC(G, 0x14): { // ZbcGetActiveSlotMask
            return Fixed<std::array<u32, 2>>(b, [](std::array<u32, 2>& p) { p = {7, 1}; return kNvSuccess; });
        }
        case GC(G, 0x15): return Fixed<u32>(b, [](u32& load) { load = 100; return kNvSuccess; }); // PmuGetGpuLoad
        case GC(G, 0x1C): { // GetGpuTime
            return Fixed<std::array<u64, 2>>(b, [](std::array<u64, 2>& p) {
                p[0] = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count());
                return kNvSuccess;
            });
        }
        }
        return Unknown(1, gc);
    }

    u32 Ioctl3(u32 gc, IoctlBufs& b) override {
        switch (gc) {
        case GC('G', 0x05): {
            return FixedInlOut<Characteristics, GpuCharacteristics>(
                b, [](Characteristics& p, std::vector<GpuCharacteristics>& out) {
                    p.buf_size = sizeof(GpuCharacteristics);
                    p.buf_addr = 0xdeadbeef;
                    p.gc = kGm20b;
                    if (!out.empty()) out[0] = kGm20b;
                    return kNvSuccess;
                });
        }
        case GC('G', 0x06): {
            return FixedInlOut<TpcMasks, u32>(b, [](TpcMasks& p, std::vector<u32>& out) {
                if (p.mask_buf_size) p.tpc_mask = 3;
                if (!out.empty()) out[0] = p.tpc_mask;
                return kNvSuccess;
            });
        }
        }
        return Unknown(3, gc);
    }

    std::shared_ptr<KEvent> QueryEvent(u32 id) override {
        if (id == 1) return error_notifier_;
        if (id == 2) return unknown_;
        return Device::QueryEvent(id);
    }

private:
    struct Zbc {
        std::array<u32, 4> ds{}, l2{};
        u32 depth = 0, format = 0, refs = 1;
        bool operator==(const Zbc& o) const {
            return ds == o.ds && l2 == o.l2 && depth == o.depth && format == o.format;
        }
    };
    u32 ZbcSet(const u32* ds, const u32* l2, u32 depth, u32 format, u32 type) {
        Zbc e;
        e.format = format;
        if (type == 1) {
            std::copy(ds, ds + 4, e.ds.begin());
            std::copy(l2, l2 + 4, e.l2.begin());
        } else if (type == 2) {
            e.depth = depth;
        } else {
            return kNvBadParameter;
        }
        auto& t = type == 1 ? colors_ : depths_;
        if (auto it = std::find(t.begin(), t.end(), e); it != t.end()) {
            ++it->refs;
        } else {
            t.push_back(e);
        }
        return kNvSuccess;
    }
    std::vector<Zbc> colors_, depths_;
    std::shared_ptr<KEvent> error_notifier_ = NewEvent("ctrl-gpu error notifier");
    std::shared_ptr<KEvent> unknown_ = NewEvent("ctrl-gpu unknown event");
};

// /dev/nvhost-gpu (one per channel) -----------------------------------------

Tegra::CommandHeader FenceAction(Tegra::Engines::Puller::FenceOperation op, u32 syncpoint) {
    Tegra::Engines::Puller::FenceAction a{};
    a.op.Assign(op);
    a.syncpoint_id.Assign(syncpoint);
    return {a.raw};
}

/// Command lists the driver appends around a submission (drippu nvhost_gpu.cpp).
Tegra::CommandList WaitCommandList(NvFence f) {
    using namespace Tegra;
    return CommandList{boost::container::small_vector<CommandHeader, 512>{
        BuildCommandHeader(BufferMethods::SyncpointPayload, 1, SubmissionMode::Increasing),
        {f.value},
        BuildCommandHeader(BufferMethods::SyncpointOperation, 1, SubmissionMode::Increasing),
        FenceAction(Engines::Puller::FenceOperation::Acquire, static_cast<u32>(f.id)),
    }};
}

Tegra::CommandList IncrementCommandList(NvFence f, bool wait_for_idle) {
    using namespace Tegra;
    boost::container::small_vector<CommandHeader, 512> l;
    if (wait_for_idle) {
        l.push_back(BuildCommandHeader(BufferMethods::WaitForIdle, 1, SubmissionMode::Increasing));
        l.push_back({});
    }
    l.push_back(BuildCommandHeader(BufferMethods::SyncpointPayload, 1, SubmissionMode::Increasing));
    l.push_back({});
    for (int i = 0; i < 2; ++i) {
        l.push_back(BuildCommandHeader(BufferMethods::SyncpointOperation, 1, SubmissionMode::Increasing));
        l.push_back(FenceAction(Engines::Puller::FenceOperation::Increment, static_cast<u32>(f.id)));
    }
    return CommandList{std::move(l)};
}

class NvHostGpu : public Device {
public:
    explicit NvHostGpu(std::string p)
        : Device(std::move(p)), channel_state(Sys().GPU().AllocateChannel()),
          syncpoint_(Syncpoints().Allocate(false)) {}
    ~NvHostGpu() override { Syncpoints().Free(syncpoint_); }

    std::shared_ptr<Tegra::Control::ChannelState> channel_state;
    std::shared_ptr<Tegra::MemoryManager> gmmu; // keeps the bound address space alive

    struct SubmitGpfifo {
        u64 address;
        u32 num_entries, flags;
        NvFence fence;
    };
    static_assert(sizeof(SubmitGpfifo) == 0x18);

    u32 Ioctl1(u32 gc, IoctlBufs& b) override {
        constexpr u32 H = 'H', G = 'G';
        switch (gc) {
        case GC(0, 0x03): { // GetWaitbase
            return Fixed<std::array<u32, 2>>(b, [](std::array<u32, 2>& p) { p[1] = 0; return kNvSuccess; });
        }
        case GC(H, 0x01): return Fixed<s32>(b, [&](s32& fd) { nvmap_fd_ = fd; return kNvSuccess; }); // SetNvmapFd
        case GC(H, 0x03): return Fixed<u32>(b, [](u32&) { return kNvSuccess; }); // SetTimeout
        case GC(H, 0x08): case GC(H, 0x1B): { // SubmitGpfifo (0x1B: kickoff, entries in guest memory)
            const bool kickoff = gc == GC(H, 0x1B);
            return FixedVariable<SubmitGpfifo, u64>(b, [&](SubmitGpfifo& p, std::vector<u64>& inline_entries) {
                std::vector<u64> entries(p.num_entries);
                if (kickoff) {
                    g_mem.Read(p.address, entries.data(), entries.size() * sizeof(u64));
                } else {
                    if (p.num_entries > inline_entries.size()) return kNvBadParameter;
                    std::copy_n(inline_entries.begin(), p.num_entries, entries.begin());
                }
                return Submit(p, entries);
            });
        }
        case GC(H, 0x09): { // AllocObjCtx
            struct P { u32 class_num, flags; u64 obj_id; };
            return Fixed<P>(b, [&](P& p) {
                static constexpr u32 kClasses[] = {0x902D, 0xB197, 0xB1C0, 0xA140, 0xB0B5, 0xB06F};
                const auto it = std::find(std::begin(kClasses), std::end(kClasses), p.class_num);
                if (it == std::end(kClasses)) return kNvBadParameter;
                const size_t i = it - std::begin(kClasses);
                if (ctx_objs_[i]) return kNvAlreadyAllocated;
                ctx_objs_[i] = true;
                p.flags = 0;
                Log("nv   gpu AllocObjCtx class 0x%x", p.class_num);
                return kNvSuccess;
            });
        }
        case GC(H, 0x0B): { // ZcullBind
            return Fixed<std::array<u64, 2>>(b, [](std::array<u64, 2>&) { return kNvSuccess; });
        }
        case GC(H, 0x0C): { // SetErrorNotifier
            return Fixed<std::array<u64, 3>>(b, [](std::array<u64, 3>&) { return kNvSuccess; });
        }
        case GC(H, 0x0D): { // SetPriority
            return Fixed<u32>(b, [&](u32& prio) {
                switch (prio) {
                case 0x32: timeslice_ = 1300; break;
                case 0x64: timeslice_ = 2600; break;
                case 0x96: timeslice_ = 5200; break;
                default: return kNvBadParameter;
                }
                return kNvSuccess;
            });
        }
        case GC(H, 0x18): case GC(H, 0x1A): { // AllocGpfifoEx, AllocGpfifoEx2
            struct P { u32 num_entries, num_jobs, flags; NvFence fence_out; u32 reserved[3]; };
            static_assert(sizeof(P) == 0x20);
            return Fixed<P>(b, [&](P& p) {
                if (channel_state->initialized) return kNvAlreadyAllocated;
                // Without an address space yet, the channel is set up on first submission.
                if (channel_state->memory_manager) Sys().GPU().InitChannel(*channel_state, gpu::ProgramId());
                p.fence_out = Syncpoints().Fence(syncpoint_);
                Log("nv   gpu AllocGpfifo %u entries, flags 0x%x -> syncpoint %u", p.num_entries, p.flags,
                    syncpoint_);
                return kNvSuccess;
            });
        }
        case GC(H, 0x1D): { // SetTimeslice
            return Fixed<u32>(b, [&](u32& t) {
                if (t < 1000 || t > 5000) return kNvBadParameter;
                timeslice_ = t;
                return kNvSuccess;
            });
        }
        case GC(G, 0x14): return Fixed<u64>(b, [&](u64& d) { user_data_ = d; return kNvSuccess; }); // SetUserData
        case GC(G, 0x15): return Fixed<u64>(b, [&](u64& d) { d = user_data_; return kNvSuccess; }); // GetUserData
        }
        return Unknown(1, gc);
    }

    u32 Ioctl2(u32 gc, IoctlBufs& b) override {
        if (gc == GC('H', 0x1B)) { // SubmitGpfifo, entries in the inline input
            return FixedInlIn<SubmitGpfifo, u64>(b, [&](SubmitGpfifo& p, std::vector<u64>& inl) {
                if (p.num_entries > inl.size()) return kNvBadParameter;
                inl.resize(p.num_entries);
                return Submit(p, inl);
            });
        }
        return Unknown(2, gc);
    }

    std::shared_ptr<KEvent> QueryEvent(u32 id) override {
        if (id >= 1 && id <= 3) return events_[id - 1];
        return Device::QueryEvent(id);
    }

private:
    // drippu SubmitGPFIFOImpl.
    u32 Submit(SubmitGpfifo& p, const std::vector<u64>& entries) {
        constexpr u32 kFenceWait = 1, kFenceIncrement = 2, kSuppressWfi = 1 << 4, kIncrementValue = 1 << 8;
        auto& gpu = Sys().GPU();
        if (!channel_state->initialized) {
            if (!channel_state->memory_manager) Fatal("nvdrv: SubmitGpfifo on a channel without an address space");
            gpu.InitChannel(*channel_state, gpu::ProgramId());
        }
        const s32 bind_id = channel_state->bind_id;
        auto& sm = Syncpoints();
        if (p.flags & kFenceWait) {
            if (p.flags & kIncrementValue) return kNvBadParameter;
            if (!sm.IsFenceSignalled(p.fence)) gpu.PushGPUEntries(bind_id, WaitCommandList(p.fence));
        }
        p.fence.id = static_cast<s32>(syncpoint_);
        const u32 increment = ((p.flags & kFenceIncrement) ? 2 : 0) + ((p.flags & kIncrementValue) ? p.fence.value : 0);
        p.fence.value = sm.IncrementMaxExt(syncpoint_, increment);

        Tegra::CommandList list(entries.size());
        std::memcpy(list.command_lists.data(), entries.data(), entries.size() * sizeof(u64));
        if (++submits_ <= 3) {
            Log("nv   gpu SubmitGpfifo #%llu: %zu entries, flags 0x%x -> fence %d/%u",
                (unsigned long long)submits_, entries.size(), p.flags, p.fence.id, p.fence.value);
        }
        gpu.PushGPUEntries(bind_id, std::move(list));
        if (p.flags & kFenceIncrement) {
            gpu.PushGPUEntries(bind_id, IncrementCommandList(p.fence, !(p.flags & kSuppressWfi)));
        }
        p.flags = 0;
        return kNvSuccess;
    }

    u32 syncpoint_;
    s32 nvmap_fd_ = 0;
    u64 user_data_ = 0;
    u32 timeslice_ = 0;
    u64 submits_ = 0;
    std::array<bool, 6> ctx_objs_{};
    std::array<std::shared_ptr<KEvent>, 3> events_{
        NewEvent("gpu SM exception breakpoint int"), NewEvent("gpu SM exception breakpoint pause"),
        NewEvent("gpu error notifier")};
};

// /dev/nvhost-as-gpu --------------------------------------------------------

class NvHostAsGpu : public Device {
public:
    using Device::Device;

    u32 Ioctl1(u32 gc, IoctlBufs& b) override {
        constexpr u32 A = 'A';
        switch (gc) {
        case GC(A, 0x01): { // BindChannel
            return Fixed<s32>(b, [&](s32& fd) {
                auto ch = DeviceAt<NvHostGpu>(fd);
                if (!ch) Fatal("nvdrv as BindChannel: fd %d is not a GPU channel", fd);
                ch->channel_state->memory_manager = gmmu_.get();
                ch->gmmu = gmmu_;
                Log("nv   as BindChannel fd %d", fd);
                return kNvSuccess;
            });
        }
        case GC(A, 0x02): { // AllocSpace
            struct P { u32 pages, page_size, flags, pad; u64 offset; };
            return Fixed<P>(b, [&](P& p) { return AllocSpace(p.pages, p.page_size, p.flags, p.offset); });
        }
        case GC(A, 0x03): { // FreeSpace
            struct P { u64 offset; u32 pages, page_size; };
            return Fixed<P>(b, [&](P& p) { return FreeSpace(p.offset, p.pages, p.page_size); });
        }
        case GC(A, 0x05): { // UnmapBuffer
            return Fixed<u64>(b, [&](u64& offset) { return UnmapBuffer(offset); });
        }
        case GC(A, 0x06): { // MapBufferEx
            struct P { u32 flags, kind, handle, page_size; s64 buffer_offset; u64 mapping_size; s64 offset; };
            static_assert(sizeof(P) == 0x28);
            return Fixed<P>(b, [&](P& p) {
                u64 off = static_cast<u64>(p.offset);
                const u32 r = MapBufferEx(p.flags, p.kind, p.handle, static_cast<u64>(p.buffer_offset),
                                          p.mapping_size, off);
                p.offset = static_cast<s64>(off);
                return r;
            });
        }
        case GC(A, 0x08): return Fixed<VaRegions>(b, [&](VaRegions& p) { return GetVaRegions(p); });
        case GC(A, 0x09): { // AllocAsEx
            struct P { u32 flags; s32 as_fd; u32 big_page_size, reserved; u64 va_start, va_end, va_split; };
            static_assert(sizeof(P) == 0x28);
            return Fixed<P>(b, [&](P& p) { return AllocAsEx(p.big_page_size, p.va_start, p.va_end, p.va_split); });
        }
        case GC(A, 0x14): { // Remap
            auto entries = ReadArray<RemapEntry>(b.in, 0);
            const u32 r = Remap(entries);
            WriteBack(b.out, 0, entries.data(), entries.size() * sizeof(RemapEntry));
            return r;
        }
        }
        return Unknown(1, gc);
    }

    u32 Ioctl3(u32 gc, IoctlBufs& b) override {
        if (gc == GC('A', 0x08)) {
            return FixedInlOut<VaRegions, VaRegion>(b, [&](VaRegions& p, std::vector<VaRegion>& out) {
                const u32 r = GetVaRegions(p);
                for (size_t i = 0; i < std::min<size_t>(2, out.size()); ++i) out[i] = p.regions[i];
                return r;
            });
        }
        return Unknown(3, gc);
    }

private:
    static constexpr u32 kSupportedBigPages = 0x30000, kDefaultBigPage = 0x20000, kVaStartShift = 10;
    static constexpr u32 kMapFixed = 1, kMapSparse = 2, kMapRemap = 1 << 8;

    struct VaRegion {
        u64 offset;
        u32 page_size, pad;
        u64 pages;
    };
    static_assert(sizeof(VaRegion) == 0x18);
    struct VaRegions {
        u64 buf_addr;
        u32 buf_size, reserved;
        VaRegion regions[2];
    };
    static_assert(sizeof(VaRegions) == 0x40);
    struct RemapEntry {
        u16 flags, kind;
        u32 handle, handle_offset_big_pages, as_offset_big_pages, big_pages;
    };
    static_assert(sizeof(RemapEntry) == 0x14);

    struct Mapping {
        DAddr ptr; // SMMU address of the mapped memory
        u64 offset, size;
        u32 handle;
        bool fixed, big_page, sparse_alloc;
    };
    struct Allocation {
        std::vector<u64> mappings;
        u64 size;
        u32 page_size;
        bool sparse, big_pages;
    };

    PageAllocator& AllocatorFor(bool big) { return big ? *big_alloc_ : *small_alloc_; }
    u32 BitsFor(bool big) const { return big ? big_page_bits_ : 12; }

    u32 AllocAsEx(u32 big_page_size, u64 va_start, u64 va_end, u64 va_split) {
        if (initialised_) Fatal("nvdrv as AllocAsEx: address space initialised twice");
        if (big_page_size) {
            if (!std::has_single_bit(big_page_size) || !(big_page_size & kSupportedBigPages)) return kNvBadValue;
            big_page_size_ = big_page_size;
            big_page_bits_ = static_cast<u32>(std::countr_zero(big_page_size));
            va_start_ = static_cast<u64>(big_page_size) << kVaStartShift;
        }
        if (va_start) {
            va_start_ = va_start;
            va_split_ = va_split;
            va_end_ = va_end;
        }
        small_alloc_.emplace(static_cast<u32>(va_start_ >> 12), static_cast<u32>(va_split_ >> 12));
        big_alloc_.emplace(static_cast<u32>(va_split_ >> big_page_bits_),
                           static_cast<u32>((va_end_ - va_split_) >> big_page_bits_));
        gmmu_ = std::make_shared<Tegra::MemoryManager>(Sys(), std::bit_width(va_end_ - 1), va_split_,
                                                       big_page_bits_, 12);
        Sys().GPU().InitAddressSpace(*gmmu_);
        initialised_ = true;
        Log("nv   as AllocAsEx big pages 0x%x, va 0x%llx..0x%llx split 0x%llx", big_page_size_,
            (unsigned long long)va_start_, (unsigned long long)va_end_, (unsigned long long)va_split_);
        return kNvSuccess;
    }

    u32 AllocSpace(u32 pages, u32 page_size, u32 flags, u64& offset) {
        if (!initialised_) return kNvBadValue;
        if (page_size != kPageSize && page_size != big_page_size_) return kNvBadValue;
        const bool big = page_size != kPageSize;
        if (!big && (flags & kMapSparse)) Fatal("nvdrv as AllocSpace: sparse small pages");
        auto& alloc = AllocatorFor(big);
        const u32 bits = BitsFor(big);
        if (flags & kMapFixed) {
            alloc.AllocateFixed(static_cast<u32>(offset >> bits), pages);
        } else {
            offset = static_cast<u64>(alloc.Allocate(pages)) << bits;
            if (!offset) return kNvInsufficientMemory;
        }
        const u64 size = static_cast<u64>(pages) * page_size;
        if (flags & kMapSparse) gmmu_->MapSparse(offset, size);
        allocations_[offset] = {{}, size, page_size, (flags & kMapSparse) != 0, big};
        Log("nv   as AllocSpace 0x%x pages of 0x%x, flags 0x%x -> gpu 0x%llx", pages, page_size, flags,
            (unsigned long long)offset);
        return kNvSuccess;
    }

    bool FreeMapping(u64 offset) {
        auto it = mappings_.find(offset);
        if (it == mappings_.end()) return false;
        const Mapping m = it->second;
        if (!m.fixed) {
            const u32 page = m.big_page ? big_page_size_ : kPageSize;
            AllocatorFor(m.big_page).Free(static_cast<u32>(m.offset >> BitsFor(m.big_page)),
                                          static_cast<u32>(AlignUp(m.size, page) >> BitsFor(m.big_page)));
        }
        UnpinHandle(m.handle);
        if (m.sparse_alloc) {
            gmmu_->MapSparse(offset, m.size, m.big_page);
        } else {
            gmmu_->Unmap(offset, m.size);
        }
        mappings_.erase(it);
        return true;
    }

    u32 FreeSpace(u64 offset, u32 pages, u32 page_size) {
        if (!initialised_) return kNvBadValue;
        auto it = allocations_.find(offset);
        if (it == allocations_.end()) return kNvBadValue;
        const Allocation a = it->second;
        if (a.page_size != page_size || a.size != static_cast<u64>(pages) * page_size) return kNvBadValue;
        for (u64 m : a.mappings) {
            if (!FreeMapping(m)) return kNvBadValue;
        }
        if (a.sparse) gmmu_->Unmap(offset, a.size);
        const bool big = page_size != kPageSize;
        AllocatorFor(big).Free(static_cast<u32>(offset >> BitsFor(big)), static_cast<u32>(a.size >> BitsFor(big)));
        allocations_.erase(it);
        return kNvSuccess;
    }

    u32 Remap(const std::vector<RemapEntry>& entries) {
        if (!initialised_) return kNvBadValue;
        for (const auto& e : entries) {
            const u64 va = static_cast<u64>(e.as_offset_big_pages) << big_page_bits_;
            const u64 size = static_cast<u64>(e.big_pages) << big_page_bits_;
            auto a = allocations_.upper_bound(va);
            if (a == allocations_.begin()) return kNvBadValue;
            --a;
            if ((va - a->first) + size > a->second.size || !a->second.sparse) return kNvBadValue;
            if (!e.handle) {
                gmmu_->MapSparse(va, size, a->second.big_pages);
                continue;
            }
            auto* h = NvMap().Get(e.handle);
            if (!h) return kNvBadValue;
            const DAddr base = PinHandle(*h);
            gmmu_->Map(va, base + (static_cast<u64>(e.handle_offset_big_pages) << big_page_bits_), size,
                       Tegra::PTEKind(e.kind), a->second.big_pages);
        }
        return kNvSuccess;
    }

    u32 MapBufferEx(u32 flags, u32 kind, u32 handle, u64 buffer_offset, u64 mapping_size, u64& offset) {
        if (!initialised_) return kNvBadValue;
        if (flags & kMapRemap) { // remap a subregion of an existing mapping
            auto it = mappings_.find(offset);
            if (it == mappings_.end() || it->second.size < mapping_size) return kNvBadValue;
            gmmu_->Map(offset + buffer_offset, it->second.ptr + buffer_offset, mapping_size, Tegra::PTEKind(kind),
                       it->second.big_page);
            return kNvSuccess;
        }
        auto* h = NvMap().Get(handle);
        if (!h) return kNvBadValue;
        const DAddr device_address = PinHandle(*h) + buffer_offset;
        const u64 size = mapping_size ? mapping_size : h->orig_size;
        if (h->align % kPageSize) Fatal("nvdrv as MapBufferEx: handle %u alignment 0x%x", handle, h->align);
        const bool big = h->align % big_page_size_ == 0;
        if (flags & kMapFixed) {
            auto a = allocations_.upper_bound(offset);
            if (a == allocations_.begin()) Fatal("nvdrv as MapBufferEx: fixed map outside any allocation");
            --a;
            if ((offset - a->first) + size > a->second.size) {
                Fatal("nvdrv as MapBufferEx: fixed map past its allocation");
            }
            const bool use_big = a->second.big_pages && big;
            gmmu_->Map(offset, device_address, size, Tegra::PTEKind(kind), use_big);
            a->second.mappings.push_back(offset);
            mappings_[offset] = {device_address, offset, size, handle, true, use_big, a->second.sparse};
        } else {
            const u64 page = big ? big_page_size_ : kPageSize;
            const u32 bits = BitsFor(big);
            offset = static_cast<u64>(AllocatorFor(big).Allocate(static_cast<u32>(AlignUp(size, page) >> bits)))
                     << bits;
            if (!offset) return kNvInsufficientMemory;
            gmmu_->Map(offset, device_address, AlignUp(size, page), Tegra::PTEKind(kind), big);
            mappings_[offset] = {device_address, offset, size, handle, false, big, false};
        }
        Log("nv   as MapBufferEx handle %u +0x%llx size 0x%llx kind 0x%x flags 0x%x -> gpu 0x%llx", handle,
            (unsigned long long)buffer_offset, (unsigned long long)size, kind, flags, (unsigned long long)offset);
        return kNvSuccess;
    }

    u32 UnmapBuffer(u64 offset) {
        if (!initialised_) return kNvBadValue;
        FreeMapping(offset);
        return kNvSuccess;
    }

    u32 GetVaRegions(VaRegions& p) {
        if (!initialised_) return kNvBadValue;
        p.buf_size = 2 * sizeof(VaRegion);
        p.regions[0] = {static_cast<u64>(small_alloc_->Start()) << 12, static_cast<u32>(kPageSize), 0,
                        small_alloc_->Limit() - small_alloc_->Start()};
        p.regions[1] = {static_cast<u64>(big_alloc_->Start()) << big_page_bits_, big_page_size_, 0,
                        big_alloc_->Limit() - big_alloc_->Start()};
        return kNvSuccess;
    }

    bool initialised_ = false;
    u32 big_page_size_ = kDefaultBigPage, big_page_bits_ = 17;
    u64 va_start_ = static_cast<u64>(kDefaultBigPage) << kVaStartShift;
    u64 va_split_ = 1ULL << 34, va_end_ = 1ULL << 37;
    std::optional<PageAllocator> small_alloc_, big_alloc_;
    std::shared_ptr<Tegra::MemoryManager> gmmu_;
    std::map<u64, Mapping> mappings_;
    std::map<u64, Allocation> allocations_;
};

std::shared_ptr<Device> OpenDevice(const std::string& path) {
    if (path == "/dev/nvmap") return std::make_shared<NvMapDevice>(path);
    if (path == "/dev/nvhost-ctrl") return std::make_shared<NvHostCtrl>(path);
    if (path == "/dev/nvhost-ctrl-gpu") return std::make_shared<NvHostCtrlGpu>(path);
    if (path == "/dev/nvhost-as-gpu") return std::make_shared<NvHostAsGpu>(path);
    if (path == "/dev/nvhost-gpu") return std::make_shared<NvHostGpu>(path);
    return nullptr;
}

// ---------------------------------------------------------------------------
// The nvdrv service.

class NvDrv : public Service {
public:
    explicit NvDrv(const char* n) : Service(n) {
        Register(0, "Open", [](IpcContext& ctx) {
            static s32 next_fd = 1;
            const std::string path = ctx.ReadInString(0);
            auto dev = OpenDevice(path);
            if (!dev) Fatal("nvdrv Open(\"%s\"): device not implemented", path.c_str());
            const s32 fd = next_fd++;
            Fds()[fd] = dev;
            Log("nv   Open(\"%s\") -> fd %d", path.c_str(), fd);
            ctx.Push<s32>(fd);
            ctx.Push<u32>(kNvSuccess);
        });
        Register(1, "Ioctl", [](IpcContext& ctx) { DoIoctl(ctx, 1); });
        Register(2, "Close", [](IpcContext& ctx) {
            Fds().erase(ctx.Pop<s32>());
            ctx.Push<u32>(kNvSuccess);
        });
        Stub(3, "Initialize", 4); // NvResult Success
        Register(4, "QueryEvent", [](IpcContext& ctx) {
            const s32 fd = ctx.Pop<s32>();
            const u32 id = ctx.Pop<u32>();
            auto it = Fds().find(fd);
            if (it == Fds().end()) Fatal("nvdrv QueryEvent on unknown fd %d", fd);
            auto ev = it->second->QueryEvent(id);
            Log("nv   QueryEvent(%s, %u)", it->second->path.c_str(), id);
            ctx.out_copy.push_back(g_handles.Add(ev));
            ctx.Push<u32>(kNvSuccess);
        });
        Stub(8, "SetClientPID", 4);
        Register(11, "Ioctl2", [](IpcContext& ctx) { DoIoctl(ctx, 2); });
        Register(12, "Ioctl3", [](IpcContext& ctx) { DoIoctl(ctx, 3); });
        Stub(13, "SetGraphicsFirmwareMemoryMarginEnabled");
    }

private:
    static void DoIoctl(IpcContext& ctx, int kind) {
        const s32 fd = ctx.Pop<s32>();
        const u32 request = ctx.Pop<u32>();
        auto it = Fds().find(fd);
        if (it == Fds().end()) Fatal("nvdrv Ioctl on unknown fd %d", fd);
        IoctlBufs b;
        b.in = ctx.ReadIn(0);
        if (kind == 2) b.inl_in = ctx.ReadIn(1);
        b.out.resize(ctx.Out(0).size);
        if (kind == 3) b.inl_out.resize(ctx.Out(1).size);
        const u32 gc = request & 0xFFFF; // group << 8 | cmd
        Device& dev = *it->second;
        const u32 result = kind == 1 ? dev.Ioctl1(gc, b) : kind == 2 ? dev.Ioctl2(gc, b) : dev.Ioctl3(gc, b);
        if (result != kNvSuccess) {
            Log("nv   %s ioctl%d 0x%08x -> NvResult 0x%x", dev.path.c_str(), kind, request, result);
        }
        if (request >> 31) { // is_out
            if (!b.out.empty()) ctx.WriteOut(0, b.out.data(), b.out.size());
            if (!b.inl_out.empty()) ctx.WriteOut(1, b.inl_out.data(), b.inl_out.size());
        }
        ctx.Push<u32>(result);
    }
};

const bool registered = [] {
    for (const char* n : {"nvdrv", "nvdrv:a", "nvdrv:s", "nvdrv:t"}) {
        RegisterServiceFactory(n, [n] { return std::make_shared<NvDrv>(n); });
    }
    return true;
}();

} // namespace

bool NvMapLookup(u32 id, u64& address, u64& size) {
    auto* h = NvMap().Get(id);
    if (!h) return false;
    address = h->address;
    size = h->size;
    return true;
}

u64 NvMapPin(u32 id) {
    auto* h = NvMap().Get(id);
    if (!h || !h->allocated) return 0;
    return PinHandle(*h);
}

void NvMapUnpin(u32 id) {
    UnpinHandle(id);
}

u64 PinHostRegion(u64 va, u64 size) {
    g_mem.MakeDeviceVisible(va, size);
    auto& smmu = Sys().Host1x().MemoryManager();
    const DAddr d = smmu.Allocate(AlignUp(size, kSmmuBigPage));
    if (!d) Fatal("nvmap: SMMU address space exhausted");
    smmu.Map(d, va, size, ProcessAsid(), true);
    return d;
}

} // namespace rt
