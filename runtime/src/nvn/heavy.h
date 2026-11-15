// RT_PROFILE_HEAVY: what a heavy submission's translation and GPU time are made of.
//
// RT_PROFILE_HEAVY=<ms> (1: 12 ms) with RT_TRACE_WAITS: every traced submission is profiled while it is
// translated - CPU time per record, the draw / dispatch phases of RT_PROFILE_DRAWS (also turned on, without its
// log), the time inside each Vulkan function (the device's function table wrapped: InstallDriverTimers), whether a
// render pass is open after each record, and per dispatch the bytes its reads and writes copied. When it retires,
// one whose translation or GPU time reached <ms> is written out (in the run folder, as waits.csv):
//   heavy_subs.csv        one row per measured item: sub,t_ms,kind,name,ms,count
//                         (kind: sub = totals; phase.draw / phase.dispatch / phase.other = a phase's CPU time,
//                         part.<kind> = a part of one; driver = one Vulkan function; gpu = shaders / rest;
//                         reuse.all / reuse.same = per draw phase, the draws checked / those whose inputs equal
//                         the previous draw's in the same command buffer, with the phase's time
//                         in them)
//   heavy_stretches.csv   one row per stretch of kCheckpointRecords records: CPU end, GPU time, open pass, ops
//   heavy_dispatches.csv  one row per dispatch: CPU / driver / GPU time, bytes in, written back and changed
// The submissions below <ms> are summed per second of uptime into heavy_subs.csv (sub = s<second>), the
// same columns, for comparison.
#pragma once

#include "nvn.h"

#include <array>
#include <vector>

namespace Vulkan::vk {
struct DeviceDispatch;
}

namespace rt::nvn {

/// RT_PROFILE_HEAVY: on, and the threshold in ms (0 when off).
double HeavyMs();
inline bool HeavyOn() { return HeavyMs() > 0; }

constexpr u32 kHeavyPhases = 11, kHeavyParts = 18; // draws.cpp's Phase and Part
// The parts pipeline.cpp (PipelineFor) and render.cpp (BindTargets) time (draws.cpp's Part values).
constexpr u32 kHeavyPartLookup = 6, kHeavyPartKey = 10, kHeavyPartChecks = 11;
/// Times a part of the current draw phase from outside draws.cpp: start (0 when the timers are off) ... end.
u64 HeavyPartStart();
void HeavyPartEnd(u32 part, u64 start);

/// per draw phase, the draws checked and those whose inputs equal the previous draw's in
/// the same command buffer (draws.cpp NoteReuse), each with the phase's time in those draws.
// After the phases, the constant buffers' check once more in parts: c0 equal apart from the draw's own words (and
// everything else as "cbufs"), then c0 alone, c2 alone, the compiler constants and uniform buffers alone; last, whether
// no GPU write was recorded in the submission since the previous draw (the keys compare places, not contents: a
// reuse of copied guest memory also needs no write over it).
constexpr u32 kHeavyReuse = kHeavyPhases + 5;
const char* HeavyReuseName(u32 slot); // heavy.cpp
// the guest memory reads of a submission's draws, by kind (uniform buffer, vertex, index,
// texel buffer, storage buffer), each a count and bytes: all; repeats of an (address, size) read earlier in the
// submission with no GPU write recorded over its 64 KB blocks since ("clean": a memo would serve it) or with one
// ("written"); and uniform reads whose 64 KB window was read before and is clean ("window": a window copy serves it).
constexpr u32 kHeavyReadKinds = 5, kHeavyReadStats = 8;
const char* HeavyReadKindName(u32 kind);  // heavy.cpp
const char* HeavyReadStatName(u32 stat);  // heavy.cpp
struct HeavyReuse {
    std::array<u64, kHeavyReuse> all_n{}, same_n{}, all_ns{}, same_ns{};
    std::array<std::array<u64, kHeavyReadStats>, kHeavyReadKinds> reads{}; // [kind][n, bytes, clean n/bytes, written n/bytes, window n/bytes]
};
extern thread_local HeavyReuse* t_heavy_reuse; // the current submission's (queue.cpp), as t_heavy_phases
constexpr u32 kHeavyKinds = 3;                    // draw, dispatch, other records
constexpr u32 kDriverFns = 200;                   // the device function table's entries, at most

/// The time inside Vulkan functions of the thread that has one installed (t_driver), per function.
struct DriverTimes {
    std::array<u64, kDriverFns> ns{}, n{};
    u64 total_ns = 0, total_n = 0;
};
extern thread_local DriverTimes* t_driver;

/// Wraps the device's function table (each entry times its call when the calling thread has t_driver); called
/// once, right after the device is made (device.cpp), when RT_PROFILE_HEAVY is on.
void InstallDriverTimers(Vulkan::vk::DeviceDispatch& d);
const char* DriverFnName(u32 id);
u32 DriverFnCount();
/// What one timed call costs on its own (two clock reads), measured at install, in ns.
double DriverTimerNs();

/// Phase (and part) CPU times of one kind of record: [phase][0] the phase, [phase][1 + part] its parts.
using HeavyPhases = std::array<std::array<u64, kHeavyParts + 1>, kHeavyPhases>;
extern thread_local HeavyPhases* t_heavy_phases; // the current record's kind; draws.cpp's timers add to it
const char* HeavyPhaseName(u32 phase);           // draws.cpp
const char* HeavyPartName(u32 part);             // draws.cpp

/// One dispatch record of a heavy submission.
struct HeavyDispatch {
    u32 record = 0;     // in the part
    u64 hash = 0;       // compute code (as the pipeline lines print it)
    u64 write_fills = 0, read_fills = 0, gpu_parts = 0, to_device = 0, from_device = 0; // TraceBytes deltas
    u32 writes = 0;     // storage / buffer-texture writes (ShaderWrite)
    u32 wb_begin = 0, wb_end = 0; // its write-backs in the submission's list
    s32 gpu_slot = -1;  // its timed dispatch (two timestamps), -1: untimed
};

/// The profile of one submission (part), filled while it is translated (queue.cpp).
struct HeavyProfile {
    bool on = false;
    u64 translate_ns = 0, t_locked_ns = 0;
    std::vector<u32> rec_ns, rec_driver_ns; // per record
    std::vector<u16> rec_driver_n;
    std::vector<u8> pass_open; // after each record
    std::array<HeavyPhases, kHeavyKinds> phases{};
    HeavyReuse reuse;
    DriverTimes driver;
    std::vector<HeavyDispatch> dispatches;
    void Reset(bool traced, size_t records); // on: RT_PROFILE_HEAVY and a traced submission
};

/// A write-back of a heavy submission at retirement: its bytes against guest memory just before it is applied.
struct HeavyWrite {
    u64 size = 0, changed = 0, first = 0, last = 0; // changed bytes, the first and last changed offset
};

/// Writes a retired submission's profile (heavy: its rows; else: into the second's sums). `gpu_ms`: its GPU
/// time; `stretch_gpu_ms`: per stretch in record order; `dispatch_gpu_ms`: per timed dispatch slot;
/// `ops`: its record types (draw 0 / dispatch 1 / other 2); `writes`: its ring write-backs' comparison (heavy only).
void HeavyRetired(u32 queue, u64 value, size_t first_record, double gpu_ms, const HeavyProfile& p,
                  const std::vector<double>& stretch_gpu_ms, u32 stretch_records, const std::vector<double>& dispatch_gpu_ms,
                  const std::vector<u8>& kinds, const std::vector<HeavyWrite>& writes);

} // namespace rt::nvn
