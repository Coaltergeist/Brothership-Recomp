// NVNcommandBuffer: command and control memory, recording and command lists.
//
// The game gives a command buffer two kinds of memory and an out-of-memory
// callback. NVN writes GPU methods into command memory (a range of a memory
// pool) and the list's structure into control memory (plain CPU memory). The
// layer writes its own records (nvn.h: Op, RecordHeader) into command memory
// and uses control memory exactly as NVN 12.3.2 does (its nvnCommandBuffer*
// functions, Ghidra 0x55d9d0-0x55e260 in the BRO SDK), so the game's callbacks
// for control memory run when the console's would; those for command memory
// run when the layer's records fill it, not NVN's methods. Control memory:
//
//   list header  16 bytes, at BeginRecording; the handle points behind it
//   group         8 bytes {kind 3, count}, followed by `count` segments
//   segment      24 bytes {address, size}: a filled range of command memory;
//                one that continues the previous segment extends it instead
//   jump         16 bytes {kind 4, next}: the game added control memory while
//                recording (the last 16 bytes of each chunk are kept for it)
//   end           8 bytes {kind 2}, at EndRecording
//   record        a call NVN handles on the CPU at submit (a "control
//                 command": syncs, pool binds, compute binds and dispatches) takes
//                 NVN's size for it in control memory, after the open
//                 segment; the layer writes its record there {kind 0x80, 3
//                 bytes, RecordHeader, payload} (ControlRecord)
//
// Callbacks as NVN's: event 0 (command memory) asks for max(needed, 512)
// bytes, event 1 (control memory) for max(needed + 16, 256); a callback that
// adds too little stops the run with NVN's message. Control commands NVN
// writes that the layer does not (yet): BindProgram's kind 18 (a program
// property), BindBlendState's kind 22, SetRenderEnableConditional's kind 21,
// and Barrier(ORDER_INDIRECT_DATA)'s segment splits. They change when the
// game's callback runs, not what the list does. Nothing here calls Vulkan: submission
// reads the lists (ForEachRecord) on the submitting thread.
#include "nvn.h"

#include <algorithm>
#include <unordered_map>

namespace rt::nvn {

namespace {

enum Event : u32 { kOutOfCommandMemory = 0, kOutOfControlMemory = 1 };
enum Kind : u8 { kListHeader = 1, kEnd = 2, kGroup = 3, kJump = 4, kRecord = 0x80 };

struct ListHeader {
    u8 kind;
    u8 pad[7];
    u64 magic;
};
struct Group {
    u8 kind;
    u8 pad[3];
    u32 count;
};
struct ControlSegment {
    u64 address;
    u32 size;
    u32 pad;
    u64 reserved;
};
struct Jump {
    u8 kind;
    u8 pad[7];
    u64 next;
};
struct End {
    u8 kind;
    u8 pad[7];
};
struct ControlRecordHeader {
    u8 kind;
    u8 pad[3];
    RecordHeader record; // its size counts from here: the entry's size - 4
};
static_assert(sizeof(ListHeader) == 16 && sizeof(Group) == 8 && sizeof(ControlSegment) == 24 &&
              sizeof(Jump) == 16 && sizeof(End) == 8);
constexpr u64 kListMagic = 0x5453494C4E564E4C; // "LNVNLIST"
constexpr u64 kKept = sizeof(Jump);            // kept free at the end of each control chunk

struct CommandBuffer : Object {
    static constexpr Type kType = Type::kCommandBuffer;
    u64 callback = 0, callback_data = 0;
    bool callback_enabled = true;
    bool recording = false;
    // Command memory: the current chunk [cmd_start, cmd_end), written up to
    // cmd_cur; [seg_start, cmd_cur) is not yet listed in control memory.
    u64 cmd_start = 0, cmd_cur = 0, cmd_end = 0, seg_start = 0;
    // Control memory: the current chunk [ctl_start, ctl_end + kKept), written up to ctl_cur.
    u64 ctl_start = 0, ctl_cur = 0, ctl_end = 0;
    u64 group = 0;  // the open group (0: the next segment starts one)
    u64 header = 0; // the recording list's header
};

// NVN's Finalize does nothing, so a game may initialise a command buffer's
// storage again without finalising it: the host object found here is then freed.
std::mutex g_live_mu;
std::unordered_map<u64, CommandBuffer*> g_live; // the game's NVNcommandBuffer -> host object

const char* MemoryName(u32 event) { return event == kOutOfCommandMemory ? "command" : "control"; }

void RunCallback(GuestContext* c, CommandBuffer* cb, u32 event, u64 min_size) {
    if (!cb->callback) Fatal("nvn: CommandBuffer has run out of %s memory with no out-of-memory callback.", MemoryName(event));
    hle::CallGuest(c, cb->callback, {cb->guest, event, min_size, cb->callback_data});
}

/// Room for `need` bytes of control memory before the kept jump space; after
/// a callback, at least `min_free` bytes free (NVN's checks).
void ControlRoom(GuestContext* c, CommandBuffer* cb, u64 need, u64 min_free) {
    if (cb->ctl_cur + need <= cb->ctl_end) return;
    RunCallback(c, cb, kOutOfControlMemory, std::max<u64>(min_free, 256));
    if (cb->ctl_end + kKept - cb->ctl_cur < min_free) {
        Fatal("nvn: CommandBuffer out-of-memory callback didn't add enough control memory.");
    }
}

/// Lists [seg_start, cmd_cur) in control memory.
void FlushSegment(GuestContext* c, CommandBuffer* cb) {
    if (cb->cmd_cur == cb->seg_start) return;
    const u64 address = cb->seg_start, size = cb->cmd_cur - cb->seg_start;
    cb->seg_start = cb->cmd_cur;
    if (cb->group) {
        const u64 last = cb->ctl_cur - sizeof(ControlSegment);
        auto s = Load<ControlSegment>(last);
        if (s.address + s.size == address && s.size + size <= 0xFFFF'FFFF) {
            s.size += static_cast<u32>(size);
            Store(last, s);
            return;
        }
    }
    ControlRoom(c, cb, sizeof(Group) + sizeof(ControlSegment), 0x30);
    if (!cb->group) { // also after a callback that added control memory
        cb->group = cb->ctl_cur;
        Store(cb->group, Group{kGroup, {}, 0});
        cb->ctl_cur += sizeof(Group);
    }
    auto g = Load<Group>(cb->group);
    ++g.count;
    Store(cb->group, g);
    Store(cb->ctl_cur, ControlSegment{address, static_cast<u32>(size), 0, 0});
    cb->ctl_cur += sizeof(ControlSegment);
}

u8* RecordBytes(Op op, const void* payload, u32 size, u32 total, u32 at, u8 (&small)[256], std::vector<u8>& large) {
    u8* bytes = small;
    if (total > sizeof small) {
        large.resize(total);
        bytes = large.data();
    }
    std::memset(bytes, 0, total);
    const RecordHeader h{op, static_cast<u16>(total - at)};
    std::memcpy(bytes + at, &h, sizeof h);
    std::memcpy(bytes + at + sizeof h, payload, size);
    return bytes;
}

} // namespace

void Record(GuestContext* c, u64 cmdbuf, Op op, const void* payload, u32 size) {
    CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    if (!cb->recording) Fatal("nvn: %s recorded outside BeginRecording/EndRecording", OpName(op));
    const u32 total = (static_cast<u32>(sizeof(RecordHeader)) + size + 3) & ~3u;
    if (total > 0xFFFF) Fatal("nvn: a %s record of %u bytes is too large", OpName(op), size);
    if (cb->cmd_cur + total > cb->cmd_end) {
        if (!cb->callback_enabled) Fatal("nvn: out of command memory with its callback disabled: not supported");
        RunCallback(c, cb, kOutOfCommandMemory, std::max<u64>(total, 512));
        if (cb->cmd_cur + total > cb->cmd_end) {
            Fatal("nvn: CommandBuffer out-of-memory callback didn't add enough command memory.");
        }
    }
    u8 small[256];
    std::vector<u8> large;
    g_mem.Write(cb->cmd_cur, RecordBytes(op, payload, size, total, 0, small, large), total);
    cb->cmd_cur += total;
}

void ControlRecord(GuestContext* c, u64 cmdbuf, Op op, const void* payload, u32 size, u32 nvn_size) {
    CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    if (!cb->recording) Fatal("nvn: %s recorded outside BeginRecording/EndRecording", OpName(op));
    if (nvn_size % 8 || sizeof(ControlRecordHeader) + size > nvn_size) {
        Fatal("nvn: a %s record of %u bytes does not fit NVN's %u", OpName(op), size, nvn_size);
    }
    FlushSegment(c, cb);
    ControlRoom(c, cb, nvn_size, nvn_size + kKept);
    u8 small[256];
    std::vector<u8> large;
    u8* bytes = RecordBytes(op, payload, size, nvn_size, 4, small, large);
    bytes[0] = kRecord;
    g_mem.Write(cb->ctl_cur, bytes, nvn_size);
    cb->ctl_cur += nvn_size;
    cb->group = 0;
}

std::vector<Segment> Segments(u64 handle) {
    const auto h = Load<ListHeader>(handle - sizeof(ListHeader));
    if (h.kind != kListHeader || h.magic != kListMagic) {
        Fatal("nvn: 0x%llx is not a command handle of the layer", (unsigned long long)handle);
    }
    std::vector<Segment> v;
    for (u64 at = handle;;) {
        switch (hle::ReadGuest<u8>(at)) {
        case kEnd: return v;
        case kJump: at = Load<Jump>(at).next; break;
        case kRecord: { // one record, read where it is
            const u64 size = Load<ControlRecordHeader>(at).record.size;
            v.push_back({at + 4, size});
            at += 4 + size;
            break;
        }
        case kGroup: {
            const u32 n = Load<Group>(at).count;
            at += sizeof(Group);
            for (u32 i = 0; i < n; ++i, at += sizeof(ControlSegment)) {
                const auto s = Load<ControlSegment>(at);
                v.push_back({s.address, s.size});
            }
            break;
        }
        default:
            Fatal("nvn: command list 0x%llx: bad control entry %u at 0x%llx", (unsigned long long)handle,
                  hle::ReadGuest<u8>(at), (unsigned long long)at);
        }
    }
}

// --- CommandBuffer ----------------------------------------------------------------------------

NVN_PROC(nvnCommandBufferInitialize, bool, u64 cmdbuf, u64 device) {
    std::lock_guard lock(g_live_mu);
    if (auto it = g_live.find(cmdbuf); it != g_live.end()) delete it->second;
    g_live[cmdbuf] = Create<CommandBuffer>(cmdbuf);
    return true;
}
NVN_PROC(nvnCommandBufferFinalize, void, u64 cmdbuf) {
    {
        std::lock_guard lock(g_live_mu);
        g_live.erase(cmdbuf);
    }
    Destroy<CommandBuffer>(cmdbuf);
}
NVN_PROC(nvnCommandBufferSetDebugLabel, void, u64 cmdbuf, u64 label) { Get<CommandBuffer>(cmdbuf); }
NVN_PROC(nvnCommandBufferSetMemoryCallback, void, u64 cmdbuf, u64 callback) {
    Get<CommandBuffer>(cmdbuf)->callback = callback;
}
NVN_PROC(nvnCommandBufferSetMemoryCallbackData, void, u64 cmdbuf, u64 data) {
    Get<CommandBuffer>(cmdbuf)->callback_data = data;
}
NVN_PROC(nvnCommandBufferGetMemoryCallback, u64, u64 cmdbuf) { return Get<CommandBuffer>(cmdbuf)->callback; }
NVN_PROC(nvnCommandBufferGetMemoryCallbackData, u64, u64 cmdbuf) { return Get<CommandBuffer>(cmdbuf)->callback_data; }
NVN_PROC(nvnCommandBufferSetCommandMemoryCallbackEnabled, void, u64 cmdbuf, bool enabled) {
    Get<CommandBuffer>(cmdbuf)->callback_enabled = enabled;
}

NVN_PROC(nvnCommandBufferAddCommandMemory, void, u64 cmdbuf, u64 pool, s64 offset, u64 size) {
    CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    const PoolRange p = PoolOf(pool);
    const u64 off = static_cast<u64>(offset);
    if (offset < 0 || off > p.size || size > p.size - off || (p.storage + off) % 4 || size % 4) {
        Fatal("nvn: nvnCommandBufferAddCommandMemory: 0x%llx+0x%llx outside its pool (0x%llx bytes) or not 4-aligned",
              (unsigned long long)off, (unsigned long long)size, (unsigned long long)p.size);
    }
    FlushSegment(c, cb);
    cb->cmd_start = cb->cmd_cur = cb->seg_start = p.storage + off;
    cb->cmd_end = cb->cmd_start + size;
}
NVN_PROC(nvnCommandBufferAddControlMemory, void, u64 cmdbuf, u64 memory, u64 size) {
    CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    if (size < 2 * kKept) Fatal("nvn: nvnCommandBufferAddControlMemory: 0x%llx bytes is too small", (unsigned long long)size);
    if (cb->recording && cb->ctl_start) { // the jump takes the kept space
        Store(cb->ctl_cur, Jump{kJump, {}, memory});
    }
    cb->ctl_start = cb->ctl_cur = memory;
    cb->ctl_end = memory + size - kKept;
    cb->group = 0;
}

NVN_PROC(nvnCommandBufferGetCommandMemorySize, u64, u64 cmdbuf) {
    const CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    return cb->cmd_end - cb->cmd_start;
}
NVN_PROC(nvnCommandBufferGetCommandMemoryUsed, u64, u64 cmdbuf) {
    const CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    return cb->cmd_cur - cb->cmd_start;
}
NVN_PROC(nvnCommandBufferGetCommandMemoryFree, u64, u64 cmdbuf) {
    const CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    return cb->cmd_end - cb->cmd_cur;
}
NVN_PROC(nvnCommandBufferGetControlMemorySize, u64, u64 cmdbuf) {
    const CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    return cb->ctl_end + kKept - cb->ctl_start;
}
NVN_PROC(nvnCommandBufferGetControlMemoryUsed, u64, u64 cmdbuf) {
    const CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    return cb->ctl_cur - cb->ctl_start;
}
NVN_PROC(nvnCommandBufferGetControlMemoryFree, u64, u64 cmdbuf) {
    const CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    return cb->ctl_end + kKept - cb->ctl_cur;
}

NVN_PROC(nvnCommandBufferBeginRecording, void, u64 cmdbuf) {
    CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    if (cb->recording) Fatal("nvn: nvnCommandBufferBeginRecording while recording");
    cb->recording = true;
    cb->group = 0;
    cb->seg_start = cb->cmd_cur;
    ControlRoom(c, cb, sizeof(ListHeader), 0x20);
    cb->header = cb->ctl_cur;
    Store(cb->header, ListHeader{kListHeader, {}, kListMagic});
    cb->ctl_cur += sizeof(ListHeader);
}
NVN_PROC(nvnCommandBufferEndRecording, u64, u64 cmdbuf) {
    CommandBuffer* cb = Get<CommandBuffer>(cmdbuf);
    if (!cb->recording) return 0; // as NVN
    FlushSegment(c, cb);
    ControlRoom(c, cb, sizeof(End), 0x18);
    Store(cb->ctl_cur, End{kEnd, {}});
    cb->ctl_cur += sizeof(End);
    cb->group = 0;
    cb->recording = false;
    return cb->header + sizeof(ListHeader);
}

} // namespace rt::nvn
