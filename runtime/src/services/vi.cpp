// vi:u (display service) and the IHOSBinderDriverRelay that carries Android
// IGraphicBufferProducer parcels for the layer's buffer queue.
//
// Each queued buffer is composited by video_core (gpu::Composite) on a
// presenter thread once its fences have passed, as drippu's hardware composer
// does. A queued buffer belongs to the display until then: DequeueBuffer only
// hands out free buffers and waits (without the kernel lock) when there is
// none, as the real service does. Handing a buffer back before its composite
// let the game draw the next frame into it first, and every other frame showed
// an old picture (ARCHITECTURE §5.4).
// Presented frames are captured as PNGs:
// RT_CAPTURE=<frame>[,<frame>...] (default none) writes frame_<n>.png in the
// working directory, from the renderer's output with Vulkan, otherwise from
// the guest buffer; the PNG is written on the capture thread (capture.cpp). RT_TRACE_PRESENT=1 logs each queued buffer: slot, fences,
// and where those syncpoints stood at that moment.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <cstdlib>
#include <set>
#include <sstream>
#include <thread>

#include "../gpu.h"
#include "../ipc.h"
#include "../overlay.h"
#include "../perf.h"
#include "../services.h"
#include "../threads.h"
#include "../platform/platform.h"

namespace rt {
namespace {

constexpr s32 kBinderId = 1;
constexpr u32 kGbfrMagic = 0x47424652;
// Offsets into the GBFR int payload (= libnx NvGraphicBuffer minus its 12-byte NativeHandle).
constexpr u32 kGbNvmapId = 0x4, kGbPlane0 = 0x34;
// Offsets into the whole flattened GBFR (drippu NvGraphicBuffer).
constexpr u32 kGbStride = 0x0C, kGbExternalFormat = 0x48;
// QueueBufferInput (present_report.md §3.3).
constexpr u32 kQbiCrop = 0x0C, kQbiTransform = 0x20, kQbiFence = 0x30;
constexpr u32 kPlaneWidth = 0x0, kPlaneHeight = 0x4, kPlanePitch = 0x14, kPlaneOffset = 0x1C,
              kPlaneBlockHeightLog2 = 0x24;

enum Transaction : u32 {
    kRequestBuffer = 1, kSetBufferCount = 2, kDequeueBuffer = 3, kQueueBuffer = 7,
    kCancelBuffer = 8, kQuery = 9, kConnect = 10, kDisconnect = 11, kSetPreallocatedBuffer = 14,
};

struct Slot {
    enum class State { kFree, kDequeued, kQueued }; // kQueued: until composited
    State state = State::kFree;                     // under BufferQueue::state_mu
    bool configured = false;
    u32 nvmap_id = 0, offset = 0, width = 0, height = 0, pitch = 0, block_height_log2 = 0, format = 0;
    u32 stride = 0, external_format = 0; // stride in pixels
    u64 device_address = 0;              // pinned while the slot is configured
    std::vector<u8> graphic_buffer; // as the producer flattened it (SetPreallocatedBuffer)
};

/// A queued frame on its way to the screen.
struct Presentation {
    u32 slot;
    u64 frame;
    std::vector<gpu::DisplayLayer> layers;
    std::vector<gpu::DisplayFence> fences;
    std::string capture_path; // empty: no capture
};

struct BufferQueue {
    Slot slots[16];
    u32 next_dequeue = 0;
    u64 frames = 0;
    std::set<u64> capture;
    std::shared_ptr<KEvent> buffer_event = std::make_shared<KEvent>();
    // Slot states are shared with the presenter thread, which runs without the kernel lock.
    std::mutex state_mu;
    std::condition_variable slot_freed;
    std::deque<Presentation> pending; // under state_mu
    std::condition_variable pending_cv;
    bool presenter_started = false;
    u32 in_flight = 0; // queued, not yet composited (under state_mu)
    std::condition_variable composited;
    // How many frames the game may queue before the GPU has composited them
    // (RT_FRAMES_AHEAD, default 1; 0 = no limit beyond the free buffers).
    const u32 frames_ahead = std::getenv("RT_FRAMES_AHEAD") ? std::stoul(std::getenv("RT_FRAMES_AHEAD")) : 1;

    BufferQueue() {
        buffer_event->name = "buffer available";
        buffer_event->signaled = true; // every buffer starts free; signalled again on each release
        std::istringstream in(std::getenv("RT_CAPTURE") ? std::getenv("RT_CAPTURE") : "");
        for (std::string f; std::getline(in, f, ',');) {
            if (!f.empty()) capture.insert(std::stoull(f));
        }
    }
};

BufferQueue& Queue() {
    static BufferQueue q;
    return q;
}

/// Reads an Android Parcel: header {payload_size, payload_off, objects_size, objects_off}.
class ParcelReader {
public:
    explicit ParcelReader(const std::vector<u8>& raw) {
        if (raw.size() >= 16) {
            u32 size, off;
            std::memcpy(&size, raw.data(), 4);
            std::memcpy(&off, raw.data() + 4, 4);
            if (off + size <= raw.size()) payload_.assign(raw.begin() + off, raw.begin() + off + size);
        }
    }
    s32 Int() {
        s32 v = 0;
        if (pos_ + 4 <= payload_.size()) std::memcpy(&v, payload_.data() + pos_, 4);
        pos_ += 4;
        return v;
    }
    void SkipInterfaceToken() {
        Int(); // strict-mode policy
        const s32 len = Int();
        pos_ += ((static_cast<size_t>(len) + 1) * 2 + 3) & ~size_t{3};
    }
    /// A flattened object: {s32 len, s32 fd_count, data}.
    std::vector<u8> Flattened() {
        const s32 len = Int();
        Int();
        std::vector<u8> v;
        if (len > 0 && pos_ + len <= payload_.size()) v.assign(payload_.begin() + pos_, payload_.begin() + pos_ + len);
        pos_ += (static_cast<size_t>(len) + 3) & ~size_t{3};
        return v;
    }

private:
    std::vector<u8> payload_;
    size_t pos_ = 0;
};

class ParcelWriter {
public:
    void Int(s32 v) { Bytes(&v, 4); }
    void Bytes(const void* p, size_t n) {
        const size_t at = payload_.size();
        payload_.resize(at + ((n + 3) & ~size_t{3}));
        std::memcpy(payload_.data() + at, p, n);
    }
    std::vector<u8> Finish() const {
        const u32 hdr[4] = {static_cast<u32>(payload_.size()), 16, 0, static_cast<u32>(16 + payload_.size())};
        std::vector<u8> out(16 + payload_.size());
        std::memcpy(out.data(), hdr, 16);
        std::memcpy(out.data() + 16, payload_.data(), payload_.size());
        return out;
    }

private:
    std::vector<u8> payload_;
};

template <typename T>
T Get(const std::vector<u8>& v, size_t off) {
    T x{};
    if (off + sizeof(T) <= v.size()) std::memcpy(&x, v.data() + off, sizeof(T));
    return x;
}

void WriteBufferOutput(ParcelWriter& w) {
    const u32 out[4] = {1280, 720, 0, 0}; // width, height, transformHint, numPendingBuffers
    w.Bytes(out, sizeof out);
}

/// The display is done with a slot: the producer may dequeue it again.
void ReleaseSlot(u32 slot_index) {
    auto& q = Queue();
    {
        std::lock_guard lock(q.state_mu);
        if (q.slots[slot_index].state == Slot::State::kQueued) q.slots[slot_index].state = Slot::State::kFree;
    }
    q.slot_freed.notify_all();
    q.buffer_event->signaled = true;
    WakeWaiters();
}

/// Composites queued frames in order, each once its fences have passed.
void PresenterMain() {
    auto& q = Queue();
    for (;;) {
        Presentation p;
        {
            std::unique_lock lock(q.state_mu);
            q.pending_cv.wait(lock, [&] { return !q.pending.empty(); });
            p = std::move(q.pending.front());
            q.pending.pop_front();
        }
        gpu::WaitForFences(p.fences);
        static const bool trace = std::getenv("RT_TRACE_PRESENT") != nullptr;
        if (trace) Log("vi   composite frame %llu (t=%.3f s): slot %u", (unsigned long long)p.frame, Uptime(), p.slot);
        if (gpu::Composite(p.layers, {}, p.capture_path)) {
            Log("vi   frame %llu (t=%.1f s): capturing the composited image to %s", (unsigned long long)p.frame,
                Uptime(), p.capture_path.c_str());
        }
        {
            std::lock_guard lock(q.state_mu);
            --q.in_flight;
        }
        q.composited.notify_all();
        ReleaseSlot(p.slot);
    }
}

void Present(u32 slot_index, const std::vector<u8>& input) {
    auto& q = Queue();
    if (slot_index >= 16) Fatal("QueueBuffer slot %u out of range", slot_index);
    const Slot& slot = q.slots[slot_index];
    const u64 frame = ++q.frames;
    perf::OnPresent();
    const bool capture = q.capture.count(frame) && slot.configured;
    const std::string path = "frame_" + std::to_string(frame) + ".png";
    // Without a drawing renderer, captures read the guest's buffer here.
    const bool capture_here = capture && !gpu::RendererDraws();
    if (slot.configured) {
        if (!slot.device_address) Fatal("presented slot %u has no device address", slot_index);
        const gpu::DisplayLayer layer{slot.device_address, slot.offset, slot.width, slot.height, slot.stride,
                                      slot.external_format, Get<u32>(input, kQbiTransform),
                                      Get<s32>(input, kQbiCrop), Get<s32>(input, kQbiCrop + 4),
                                      Get<s32>(input, kQbiCrop + 8), Get<s32>(input, kQbiCrop + 12)};
        // Fence: { u32 num_fences; NvFence fences[4]; }
        std::vector<gpu::DisplayFence> fences;
        const u32 num_fences = std::min<u32>(Get<u32>(input, kQbiFence), 4);
        for (u32 i = 0; i < num_fences; ++i) {
            fences.push_back({Get<s32>(input, kQbiFence + 4 + i * 8), Get<u32>(input, kQbiFence + 8 + i * 8)});
        }
        static const bool trace = std::getenv("RT_TRACE_PRESENT") != nullptr;
        if (trace) {
            std::string f;
            for (const auto& fence : fences) {
                char b[96];
                if (fence.id < 0) {
                    std::snprintf(b, sizeof b, " [id %d]", fence.id);
                } else {
                    std::snprintf(b, sizeof b, " [syncpoint %d >= %u; guest %u host %u]", fence.id, fence.value,
                                  gpu::SyncpointGuestValue(fence.id), gpu::SyncpointHostValue(fence.id));
                }
                f += b;
            }
            Log("vi   present frame %llu (t=%.3f s): slot %u, %u fence(s)%s", (unsigned long long)frame, Uptime(),
                slot_index, num_fences, f.c_str());
        }
        std::vector<gpu::DisplayLayer> layers{layer};
        if (const auto top = overlay::Layer()) layers.push_back(*top);
        {
            std::lock_guard lock(q.state_mu);
            q.slots[slot_index].state = Slot::State::kQueued;
            q.pending.push_back({slot_index, frame, std::move(layers), std::move(fences),
                                 capture && !capture_here ? path : std::string()});
            ++q.in_flight;
            if (!q.presenter_started) {
                q.presenter_started = true;
                std::thread(PresenterMain).detach();
            }
        }
        q.pending_cv.notify_one();
        // Keep the game at most `frames_ahead` frames ahead of the GPU.
        if (q.frames_ahead) {
            RunWithoutGil([&] {
                std::unique_lock lock(q.state_mu);
                q.composited.wait(lock, [&] { return q.in_flight <= q.frames_ahead; });
            });
        }
    }
    if (capture_here) {
        u64 base = 0, size = 0;
        if (!NvMapLookup(slot.nvmap_id, base, size)) {
            Fatal("presented slot %u names unknown nvmap id %u", slot_index, slot.nvmap_id);
        }
        const PresentedBuffer buf{base + slot.offset, slot.width, slot.height, slot.pitch,
                                  slot.block_height_log2, slot.format};
        Log("vi   frame %llu (t=%.1f s): capturing slot %u (%ux%u, pitch %u, format %u) to %s",
            (unsigned long long)frame, Uptime(), slot_index, slot.width, slot.height, slot.pitch, slot.format, path.c_str());
        CapturePng(buf, path);
    }
    HidOnFrame(frame);
}

std::vector<u8> Transact(u32 code, const std::vector<u8>& request) {
    ParcelReader r(request);
    r.SkipInterfaceToken();
    ParcelWriter w;
    auto& q = Queue();
    switch (code) {
    case kConnect:
        WriteBufferOutput(w);
        w.Int(0);
        break;
    case kSetPreallocatedBuffer: {
        const s32 slot_index = r.Int();
        const bool has_input = r.Int() != 0;
        if (slot_index < 0 || slot_index >= 16) {
            Fatal("SetPreallocatedBuffer slot %d out of range", slot_index);
        }
        Slot& slot = q.slots[slot_index];
        if (slot.device_address) NvMapUnpin(slot.nvmap_id);
        slot = {};
        if (has_input) {
            const std::vector<u8> gb = r.Flattened();
            if (Get<u32>(gb, 0) != kGbfrMagic) {
                Fatal("SetPreallocatedBuffer: graphic buffer magic 0x%08x", Get<u32>(gb, 0));
            }
            const std::vector<u8> ints(gb.begin() + 40, gb.end()); // after the 10-word GBFR header
            slot.configured = true;
            slot.graphic_buffer = gb;
            slot.format = Get<u32>(gb, 16);
            slot.nvmap_id = Get<u32>(ints, kGbNvmapId);
            slot.width = Get<u32>(ints, kGbPlane0 + kPlaneWidth);
            slot.height = Get<u32>(ints, kGbPlane0 + kPlaneHeight);
            slot.pitch = Get<u32>(ints, kGbPlane0 + kPlanePitch);
            slot.offset = Get<u32>(ints, kGbPlane0 + kPlaneOffset);
            slot.block_height_log2 = Get<u32>(ints, kGbPlane0 + kPlaneBlockHeightLog2);
            slot.stride = Get<u32>(gb, kGbStride);
            slot.external_format = Get<u32>(gb, kGbExternalFormat);
            slot.device_address = NvMapPin(slot.nvmap_id);
            Log("vi   slot %d: nvmap %u +0x%x (device 0x%llx), %ux%u pitch %u (stride %u px), block height 2^%u, "
                "format %u (external %u)",
                slot_index, slot.nvmap_id, slot.offset, (unsigned long long)slot.device_address, slot.width,
                slot.height, slot.pitch, slot.stride, slot.block_height_log2, slot.format, slot.external_format);
        }
        w.Int(0);
        break;
    }
    case kDequeueBuffer: {
        // The next free configured slot after the last one handed out; if the
        // display holds them all, wait for the presenter to release one.
        const auto pick = [&q]() -> int {
            for (u32 n = 0; n < 16; ++n) {
                const u32 s = (q.next_dequeue + n) % 16;
                if (q.slots[s].configured && q.slots[s].state == Slot::State::kFree) return static_cast<int>(s);
            }
            return -1;
        };
        int picked;
        {
            std::unique_lock lock(q.state_mu);
            picked = pick();
            const bool any_configured = std::any_of(std::begin(q.slots), std::end(q.slots),
                                                    [](const Slot& s) { return s.configured; });
            if (picked < 0 && any_configured) {
                lock.unlock();
                RunWithoutGil([&] {
                    std::unique_lock wait(q.state_mu);
                    q.slot_freed.wait(wait, [&] { return (picked = pick()) >= 0; });
                    q.slots[picked].state = Slot::State::kDequeued;
                });
            } else if (picked >= 0) {
                q.slots[picked].state = Slot::State::kDequeued;
            } else {
                picked = static_cast<int>(q.next_dequeue); // nothing configured yet
            }
        }
        const u32 slot = static_cast<u32>(picked);
        q.next_dequeue = (slot + 1) % 16;
        if (std::getenv("RT_TRACE_PRESENT")) Log("vi   dequeue (t=%.3f s): slot %u", Uptime(), slot);
        w.Int(static_cast<s32>(slot));
        // Always return a fence, as the real service does: libnx waits on the
        // returned NvMultiFence even when the reply leaves it out. Zero fences = ready.
        w.Int(1);
        const u8 fence[0x24] = {}; // { u32 num_fences; NvFence fences[4]; }
        w.Int(sizeof fence);
        w.Int(0); // fd count
        w.Bytes(fence, sizeof fence);
        w.Int(0);
        break;
    }
    case kRequestBuffer: {
        // The slot's GraphicBuffer, flattened: { u32 non-null; s32 size; s32 fd count; bytes }.
        // libnx ignores it; NVN's window code uses it.
        const s32 slot_index = r.Int();
        if (slot_index < 0 || slot_index >= 16) Fatal("RequestBuffer slot %d out of range", slot_index);
        const auto& gb = q.slots[slot_index].graphic_buffer;
        if (gb.empty()) {
            w.Int(0);
        } else {
            w.Int(1);
            w.Int(static_cast<s32>(gb.size()));
            w.Int(0);
            w.Bytes(gb.data(), gb.size());
        }
        w.Int(0);
        break;
    }
    case kQueueBuffer: {
        const u32 slot_index = static_cast<u32>(r.Int());
        Present(slot_index, r.Flattened()); // QueueBufferInput, 0x54 bytes
        WriteBufferOutput(w);
        w.Int(0);
        break;
    }
    case kCancelBuffer: {
        const s32 slot_index = r.Int();
        if (slot_index >= 0 && slot_index < 16) {
            {
                std::lock_guard lock(q.state_mu);
                if (q.slots[slot_index].state == Slot::State::kDequeued) q.slots[slot_index].state = Slot::State::kFree;
            }
            q.slot_freed.notify_all();
        }
        w.Int(0);
        break;
    }
    case kDisconnect:
    case kSetBufferCount:
        w.Int(0);
        break;
    case kQuery:
        w.Int(0);
        w.Int(0);
        break;
    default:
        Fatal("unimplemented IGraphicBufferProducer transaction %u", code);
    }
    return w.Finish();
}

class HosBinderDriverRelay : public Service {
public:
    HosBinderDriverRelay() : Service("vi:IHOSBinderDriverRelay") {
        auto transact = [](IpcContext& ctx) {
            const s32 id = ctx.Pop<s32>();
            const u32 code = ctx.Pop<u32>();
            if (id != kBinderId) {
                Fatal("binder transaction %u to unknown binder %d", code, id);
            }
            const std::vector<u8> request = ctx.ReadIn(0);
            const std::vector<u8> reply = Transact(code, request);
            if (g_trace_ipc) {
                Log("vi   binder transaction %u: request %zu bytes, reply %zu bytes (out buffer %llu)", code,
                    request.size(), reply.size(), (unsigned long long)ctx.Out(0).size);
            }
            ctx.WriteOut(0, reply.data(), reply.size());
        };
        Register(0, "TransactParcel", transact);
        Stub(1, "AdjustRefcount");
        Register(2, "GetNativeHandle", [](IpcContext& ctx) {
            ctx.out_copy.push_back(g_handles.Add(Queue().buffer_event));
        });
        Register(3, "TransactParcelAuto", transact);
    }
};

class ApplicationDisplayService : public Service {
public:
    ApplicationDisplayService() : Service("vi:IApplicationDisplayService") {
        Register(100, "GetRelayService", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<HosBinderDriverRelay>()); });
        Register(1010, "OpenDisplay", [](IpcContext& ctx) { ctx.Push<u64>(1); });
        Stub(1020, "CloseDisplay");
        Register(1102, "GetDisplayResolution", [](IpcContext& ctx) {
            ctx.Push<u64>(ReportDocked() ? 1920 : 1280);
            ctx.Push<u64>(ReportDocked() ? 1080 : 720);
        });
        Register(2020, "OpenLayer", [](IpcContext& ctx) {
            // Native window parcel: { u32 magic, process, binder id, pad[3]; char driver[8]; u64 pad }.
            u8 payload[0x28] = {};
            const u32 head[3] = {2, 1, static_cast<u32>(kBinderId)};
            std::memcpy(payload, head, sizeof head);
            std::memcpy(payload + 0x18, "dispdrv", 8);
            // Parcel { data_size, data_offset, objects_size, objects_offset }, the data,
            // then one (empty) object entry: the SDK's parcel reader expects it
            // (libnx does not look). As drippu's OutputParcel::WriteInterface.
            const u32 hdr[4] = {sizeof payload, 0x10, 4, 0x10 + sizeof payload};
            u8 parcel[0x10 + sizeof payload + 4] = {};
            std::memcpy(parcel, hdr, sizeof hdr);
            std::memcpy(parcel + 0x10, payload, sizeof payload);
            ctx.WriteOut(0, parcel, sizeof parcel);
            ctx.Push<u64>(sizeof parcel);
        });
        Stub(2021, "CloseLayer");
        Stub(2101, "SetLayerScalingMode");
        Register(5202, "GetDisplayVsyncEvent", [](IpcContext& ctx) {
            // Signalled at 60 Hz by a host thread; guests clear it after each wait.
            static auto vsync = [] {
                auto e = std::make_shared<KEvent>();
                e->name = "vsync";
                e->signaled = true;
                std::thread([e] {
                    auto next = std::chrono::steady_clock::now();
                    for (;;) {
                        next += std::chrono::microseconds(16667);
                        platform::SleepUntil(next);
                        std::lock_guard lock(g_gil);
                        e->signaled = true;
                        WakeWaiters(); // without it a waiter sees vsync up to 10 ms late
                    }
                }).detach();
                return e;
            }();
            ctx.out_copy.push_back(g_handles.Add(vsync));
        });
    }
};

class ViApplication : public Service {
public:
    ViApplication() : Service("vi:u") {
        Register(0, "GetDisplayService", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<ApplicationDisplayService>()); });
        // The SDK's nn::vi::Initialize uses this one for the application root:
        // { u32 policy; char proxy_name[8] } (the name has alignment 1, so it
        // follows the u32). "dispdrv" is the normal display driver; the SDK sends
        // an empty name for it on the application path.
        Register(1, "GetDisplayServiceWithProxyNameExchange", [](IpcContext& ctx) {
            char name[9] = {};
            for (int i = 0; i < 8; ++i) name[i] = ctx.At<char>(4 + i);
            if (name[0] && std::string(name) != "dispdrv") {
                std::string hex;
                for (u8 b : ctx.data) hex += std::to_string(b) + " ";
                Fatal("vi:u: display proxy \"%s\" not supported (payload %s)", name, hex.c_str());
            }
            ctx.PushObject(std::make_shared<ApplicationDisplayService>());
        });
    }
};

const bool registered = [] {
    RegisterServiceFactory("vi:u", [] { return std::make_shared<ViApplication>(); });
    return true;
}();

} // namespace
} // namespace rt
