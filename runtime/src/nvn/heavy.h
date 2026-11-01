// RT_PROFILE_HEAVY: what a heavy submission's translation and GPU time are made of.
//
// RT_PROFILE_HEAVY=<ms> (1: 12 ms) with RT_TRACE_WAITS: every traced submission is profiled while it is
// translated - CPU time per record, the draw / dispatch phases of RT_PROFILE_DRAWS (also turned on, without its
// log), the time inside each Vulkan function (the device's function table wrapped: InstallDriverTimers), whether a
// render pass is open after each record, and per dispatch the bytes its reads and writes copied. When it retires,
// one whose translation or GPU time reached <ms> is written out (in the run folder, as waits.csv):
//   heavy_subs.csv        one row per measured item: sub,t_ms,kind,name,ms,count
//                         (kind: sub = totals; phase.draw / phase.dispatch / phase.other = a phase's CPU time,
//                         part.<kind> = a part of one; driver = one Vulkan function; gpu = shaders / rest)
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

constexpr u32 kHeavyPhases = 11, kHeavyParts = 10; // draws.cpp's Phase and Part
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
