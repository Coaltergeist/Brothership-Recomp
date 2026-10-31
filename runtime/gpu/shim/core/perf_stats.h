// Shim for drippu's core/perf_stats.h: video_core reports frame ends only.
#pragma once

namespace Core {

class PerfStats {
public:
    void EndGameFrame();
};

} // namespace Core
