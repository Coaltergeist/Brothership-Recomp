// The NVN layer's internals: function registration, object headers, and calls
// into the game's own SDK (layer.h for the outside view).
#pragma once

#include "../hle.h"

#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace rt::nvn {

// --- Functions --------------------------------------------------------------------
// NVN_PROC(nvnXxx, R, A...) { ... } defines the layer's nvnXxx as a typed host
// function (hle::Bind: arguments by the AAPCS64, guest pointers as u64) and
// registers it under its name. A name outside names.inc stops the run at Init.

void RegisterProc(const char* name, hle::HostFn fn);
struct ProcRegistrar {
    ProcRegistrar(const char* name, hle::HostFn fn) { RegisterProc(name, fn); }
};

#define NVN_PROC(name, ret, ...)                                                                    \
    ret name([[maybe_unused]] GuestContext* c __VA_OPT__(, ) __VA_ARGS__);                          \
    static const ::rt::nvn::ProcRegistrar name##_registrar(#name, ::rt::hle::Bind<&name>());        \
    ret name([[maybe_unused]] GuestContext* c __VA_OPT__(, ) __VA_ARGS__)

/// The host address the layer hands out for an NVN name, or 0 for a name it
/// does not know (as the driver answers).
u64 Lookup(const std::string& name);

// --- Objects -------------------------------------------------------------------------
// Initialised objects (devices, queues, pools, textures ...) keep their state on
// the host. The game's storage for one holds a 24-byte header at offset 0: a
// magic word, the type, the object's own address, and the host object. The
// smallest initialised object of SDK 3.5.0 to 12.3.2 is at least 24 bytes
// (field accesses measured 2026-10-15). A header whose address field is not
// the object's address was copied there by the game. Builders and the small
// state objects hold their fields inline instead (builder.h, state.cpp).

enum class Type : u32 {
    kDevice = 1, kQueue, kCommandBuffer, kMemoryPool, kBuffer, kTexture, kSampler, kTexturePool,
    kSamplerPool, kProgram, kSync, kEvent, kWindow,
};
const char* TypeName(Type t);

struct Object {
    virtual ~Object() = default;
    u64 guest = 0; // the address of the game's storage
};

struct Header {
    u32 magic;
    Type type;
    u64 self;
    u64 host; // Object*
};
static_assert(sizeof(Header) == 24);
constexpr u32 kMagic = 0x4C4E564E; // "NVNL"

Object* ObjectOf(u64 guest, Type type);
/// The same, or nullptr if no initialised object of that type is there (finalised, never initialised).
Object* FindObject(u64 guest, Type type);
void PutHeader(u64 guest, Type type, Object* o);

/// The host object of the game's initialised object at `guest`; stops the run
/// if there is none of type T::kType (or if the game copied it: not supported).
template <typename T>
T* Get(u64 guest) {
    return static_cast<T*>(ObjectOf(guest, T::kType));
}

/// Creates the host object for the game's object at `guest`.
template <typename T>
T* Create(u64 guest) {
    auto* o = new T;
    o->guest = guest;
    PutHeader(guest, T::kType, o);
    return o;
}

/// Ends the game's object at `guest` (its Finalize).
template <typename T>
void Destroy(u64 guest) {
    T* o = Get<T>(guest);
    PutHeader(guest, T::kType, nullptr);
    delete o;
}

// --- GPU addresses (memory.cpp) -------------------------------------------------------
// A GPU address is the guest address of a memory pool's storage plus an offset.

struct PoolRange {
    u64 pool = 0;    // the game's NVNmemoryPool (0: no pool holds the address)
    u64 storage = 0; // its storage (= its GPU address)
    u64 size = 0;
    u64 offset = 0;  // of the address looked up
};
/// The pool holding `gpu_address`.
PoolRange FindPool(u64 gpu_address);
/// The game's initialised memory pool `pool` (offset 0); stops the run if it is none.
PoolRange PoolOf(u64 pool);
/// Tap mode: checks of memory.cpp's rules against the running driver.
void InstallMemoryShadows();
/// Tap mode: records the real driver's texture storage sizes (texture.cpp).
void InstallTextureShadows();
/// Tap mode: the guest address of the real driver's GPU address `gpu` in a
/// memory pool the shadows have seen initialised, or 0.
u64 DriverGpuToGuest(u64 gpu);
/// Tap mode: research captures of shaders and submitted command lists (research.cpp).
void InstallCaptureShadows();
/// Tap mode: checks of program.cpp's control-section parser and debug tags.
void InstallProgramShadows();

// --- Command lists (cmdbuf.cpp) ---------------------------------------------------------
// A command buffer writes the layer's own records into the game's command
// memory and lists the memory segments it filled in the game's control memory,
// consuming both as NVN does (sizes and callbacks: cmdbuf.cpp). A command
// handle is an address in control memory; the game treats it as a number.

// One entry per record type (commands.cpp): X(name).
#define NVN_OPS(X)                                                                                      \
    X(ReportCounter) X(ResetCounter) X(Barrier) X(SetTexturePool) X(SetSamplerPool) X(FenceSync) X(WaitSync)            \
    X(CopyBufferToTexture) X(ClearTexture) X(SetRenderTargets) X(SetViewports) X(SetDepthRanges) X(SetScissors)  \
    X(BindMultisampleState) X(ClearColor) X(ClearDepthStencil) X(SignalEvent)     \
    X(BindVertexAttribs) X(BindVertexStreams) X(BindProgram) X(BindState) X(SetStencil) X(SetPolygonOffsetClamp) \
    X(BindHandle) X(BindBuffer) X(Draw) X(DrawIndirect) X(ClearBuffer) X(Dispatch) X(ScratchMemory) X(SetDepthBounds)       \
    X(UpdateBuffer) X(CopyTextureToBuffer) X(CopyTextureToTexture) X(CopyBufferToBuffer) X(DispatchIndirect)

enum class Op : u16 {
    kNone,
#define NVN_OP_ENUM(name) k##name,
    NVN_OPS(NVN_OP_ENUM)
#undef NVN_OP_ENUM
};
const char* OpName(Op op);

// Payloads shared by recording and submission (the others are private to one file).
struct CounterRecord { // ReportCounter
    u32 counter;
    u32 pad;
    u64 address;
};
struct SyncRecord { // FenceSync, WaitSync: the game's NVNsync
    u64 sync;
};

struct RecordHeader {
    Op op;
    u16 size; // bytes, header included, a multiple of 4
};
static_assert(sizeof(RecordHeader) == 4);

/// Appends a record to the recording command buffer `cmdbuf` (the game's
/// NVNcommandBuffer); may run the game's out-of-memory callback.
void Record(GuestContext* c, u64 cmdbuf, Op op, const void* payload, u32 size);
template <typename T>
void Record(GuestContext* c, u64 cmdbuf, Op op, const T& payload) {
    static_assert(std::is_trivially_copyable_v<T>);
    Record(c, cmdbuf, op, &payload, sizeof payload);
}

/// Appends a record that NVN keeps in control memory (a control command,
/// cmdbuf.cpp), taking NVN's `nvn_size` bytes there as NVN does.
void ControlRecord(GuestContext* c, u64 cmdbuf, Op op, const void* payload, u32 size, u32 nvn_size);
template <typename T>
void ControlRecord(GuestContext* c, u64 cmdbuf, Op op, const T& payload, u32 nvn_size) {
    static_assert(std::is_trivially_copyable_v<T>);
    ControlRecord(c, cmdbuf, op, &payload, sizeof payload, nvn_size);
}

/// The guest memory ranges of the command list `handle`, in order: segments of
/// command memory and control records.
struct Segment {
    u64 address, size;
};
std::vector<Segment> Segments(u64 handle);

/// Calls f(op, payload, payload_size) for each record of the command list
/// `handle`, in order; the payload is a host copy.
template <typename F>
void ForEachRecord(u64 handle, F&& f) {
    std::vector<u8> bytes;
    for (const Segment& s : Segments(handle)) {
        bytes.resize(s.size);
        g_mem.Read(s.address, bytes.data(), s.size);
        for (u64 at = 0; at < s.size;) {
            RecordHeader h;
            std::memcpy(&h, bytes.data() + at, sizeof h);
            if (h.size < sizeof h || h.size % 4 || at + h.size > s.size) {
                Fatal("nvn: command list 0x%llx: bad record (op %u, size %u) at 0x%llx", (unsigned long long)handle,
                      static_cast<u32>(h.op), h.size, (unsigned long long)(s.address + at));
            }
            f(h.op, bytes.data() + at + sizeof h, h.size - sizeof h);
            at += h.size;
        }
    }
}

// --- Texture formats (formats.cpp) -----------------------------------------------------
struct FormatInfo {
    u8 bytes;                // per block
    u8 block_w, block_h;     // texels per block (1 x 1 unless compressed)
    bool depth_stencil;
};
/// The NVN format id's sizes, or nullptr for an id the layer does not know.
const FormatInfo* FindFormat(u32 id);

// --- Inline structures in the game's memory ------------------------------------------
// A builder or state object: our own layout in the game's storage. Read, change,
// write back; the size must fit the smallest storage of every SDK.

template <typename T>
T Load(u64 guest) {
    return hle::ReadGuest<T>(guest);
}
template <typename T>
void Store(u64 guest, const T& v) {
    hle::WriteGuest(guest, v);
}

// --- The game's own SDK -----------------------------------------------------------------
// Some answers depend only on the SDK the game was built with (API version,
// device limits): the layer asks the game's own NVN functions for them.

/// The game's own (lifted) function for an NVN name, or 0.
u64 RealProc(GuestContext* c, const std::string& name);

/// Guest scratch memory for calls into the SDK, with a lock: hle's scratch
/// region, divided among the users that need it.
struct Scratch {
    static constexpr u64 kStrings = kHostScratchRegion;              // 4 KB: names
    static constexpr u64 kDevice = kHostScratchRegion + 0x1000;      // 32 KB: a zeroed stand-in NVNdevice
    static constexpr u64 kOut = kHostScratchRegion + 0x9000;         // 4 KB: out parameters
    static constexpr u64 kObjects = kHostScratchRegion + 0xA000;     // 24 KB: objects for the difftest
    static constexpr u64 kCommands = kHostScratchRegion + 0x10000;   // 64 KB: command and control memory for the difftest
    static constexpr u64 kTextures = kHostScratchRegion + 0x20000;   // 512 KB: a memory pool for the difftest's textures
};
std::unique_lock<std::recursive_mutex> LockScratch();

/// nvnDeviceGetInteger as the game's SDK answers it, with a zeroed stand-in
/// device (the API version and the device limits don't read the device).
s32 SdkDeviceInteger(GuestContext* c, u32 which);
/// The game's own NVN function `name` called with the zeroed stand-in device
/// and `args`: for functions that read only their arguments (checked in the
/// SDKs for each user).
u64 SdkPure(GuestContext* c, const char* name, u64 a1 = 0, u64 a2 = 0, u64 a3 = 0);
/// The layer's nvnDeviceGetInteger: the device's values (device.cpp), else the SDK's.
s32 DeviceInteger(GuestContext* c, u32 which);

// --- Queues and syncs for the window (queue.cpp) ---------------------------------------------------
/// The value of the queue's latest submission (its work so far).
u64 QueueSubmitted(u64 queue);
/// Waits (kernel lock released) until the queue has finished submission `value`.
void WaitQueue(u64 queue, u64 value);
/// For RT_TRACE_PRESENT: milliseconds the calling threads spent translating submissions and waiting for
/// the GPU (syncs, Finish, acquire), and the number of submissions, since the previous call.
struct QueueTimes {
    double submit_ms, wait_ms;
    u64 submits;
    double gpu_max_ms; // the longest time from vkQueueSubmit to retirement among the submissions retired
};
QueueTimes TakeQueueTimes();
/// Marks the game's NVNsync signaled now, with no GPU work (until it is fenced again).
void SignalSyncOnHost(u64 sync);

/// The pipeline cache on disk (pipeline.cpp): written by a helper thread when pipelines were made since the
/// last write and a few seconds have passed. Called at each present.
void SavePipelineCacheIfChanged();
/// Reads the pipeline cache file on a thread of its own (pipeline.cpp; once, right after the device exists).
void StartPipelineCacheLoad();
/// RT_NVN_SEED_ONLY=1 (pipeline.cpp): builds RT_NVN_SEED's stages into the driver's cache, then ends the run.
void RunSeedOnly();
/// A new frame for the time draws may wait for their pipelines (pipeline.cpp, RT_NVN_PIPELINE_WAIT): at each present.
void PipelineFrameStart();
/// RT_NVN_DIFFTEST (difftest.cpp): runs at the first NVN call, then ends the run.
[[noreturn]] void RunDiffTest(GuestContext* c);

} // namespace rt::nvn
