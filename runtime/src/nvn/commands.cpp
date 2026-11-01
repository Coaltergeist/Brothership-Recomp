// The NVNcommandBuffer calls that record commands: one record
// per call (nvn.h: Op), written by cmdbuf.cpp into command memory, or into
// control memory where NVN keeps the call there (ControlRecord, with NVN's
// size for it, read from the SDK). Recording never calls Vulkan; submission
// translates the records.
#include "nvn.h"

#include <mutex>
#include <set>

namespace rt::nvn {

const char* OpName(Op op) {
    switch (op) {
#define NVN_OP_NAME(name) \
    case Op::k##name: return #name;
        NVN_OPS(NVN_OP_NAME)
#undef NVN_OP_NAME
    case Op::kNone: break;
    }
    return "?";
}

// --- Counters and barriers (FenceSync and WaitSync: queue.cpp) -------------------------------------------------------------

// ReportCounter(counter, address) and ResetCounter(counter): translated at submission (queue.cpp).
NVN_PROC(nvnCommandBufferReportCounter, void, u64 cmdbuf, u32 counter, u64 address) {
    Record(c, cmdbuf, Op::kReportCounter, CounterRecord{counter, 0, address});
}
NVN_PROC(nvnCommandBufferResetCounter, void, u64 cmdbuf, u32 counter) {
    Record(c, cmdbuf, Op::kResetCounter, CounterRecord{counter, 0, 0});
}

// Each distinct set of bits is logged once, with the guest's return address (a tail call shows its caller's):
// which barriers the game issues (bit 0x8 orders indirect data, BRO-SDK 0x5918d0).
NVN_PROC(nvnCommandBufferBarrier, void, u64 cmdbuf, u32 bits) {
    {
        static std::mutex mu;
        static std::set<u32> seen;
        std::lock_guard lock(mu);
        if (seen.insert(bits).second) {
            Log("nvn: Barrier 0x%x first seen (guest return address 0x%llx)", bits, (unsigned long long)c->x[30]);
        }
    }
    Record(c, cmdbuf, Op::kBarrier, bits);
}

// --- Texture and sampler pools -----------------------------------------------------------------
// Control commands in NVN (kinds 19 and 20, 16 bytes): the game's pool object.

NVN_PROC(nvnCommandBufferSetTexturePool, void, u64 cmdbuf, u64 pool) {
    ControlRecord(c, cmdbuf, Op::kSetTexturePool, pool, 16);
}
NVN_PROC(nvnCommandBufferSetSamplerPool, void, u64 cmdbuf, u64 pool) {
    ControlRecord(c, cmdbuf, Op::kSetSamplerPool, pool, 16);
}

} // namespace rt::nvn
