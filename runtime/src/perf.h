// Frame-rate measurement from the present path (vi QueueBuffer).
//
// Each second of wall-clock time with frames in it gives one log line
// ("perf: t=73 s, 23 fps, frame time p50/p95/max 42.1/45.3/61.0 ms") and one row
// of fps.csv in the working directory (seconds without frames give rows of 0).
// A frame's time is the gap since the previous presented frame.
// RT_PERF_LOG=0 keeps the CSV but drops the log lines.
#pragma once

#include "rt.h"

namespace rt::perf {

/// Called once per presented frame, before it is composited.
void OnPresent();

/// The last complete second: frames presented, frame times in ms (all 0 before the first one).
struct Second {
    u32 frames = 0;
    double p50_ms = 0, p95_ms = 0, max_ms = 0;
};
Second LastSecond();

/// Frames presented so far, and the Uptime() of the latest one (for the watchdog).
u64 Frames();
double LastFrameUptime();

} // namespace rt::perf
