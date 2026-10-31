#include "perf.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace rt::perf {
namespace {

struct State {
    std::mutex mu;
    std::FILE* csv = nullptr;
    bool log = true;
    double last_frame = -1; // Uptime() of the previous frame
    s64 second = -1;        // the second being collected
    std::vector<double> gaps_ms;
    Second last; // under mu
    std::atomic<u64> frames{0};
    std::atomic<double> last_frame_uptime{0};
};

State& S() {
    static State s;
    return s;
}

double Percentile(std::vector<double>& v, double p) {
    const size_t i = std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
    return v[i];
}

/// Writes the finished second `s.second` (mu held).
void Flush(State& s) {
    const u32 frames = static_cast<u32>(s.gaps_ms.size());
    s.last = {};
    if (!frames) {
        std::fprintf(s.csv, "%lld,0,,,,\n", (long long)s.second);
        return;
    }
    double sum = 0, max = 0;
    for (double g : s.gaps_ms) {
        sum += g;
        max = std::max(max, g);
    }
    const double mean = sum / frames, p50 = Percentile(s.gaps_ms, 0.50), p95 = Percentile(s.gaps_ms, 0.95);
    s.last = {frames, p50, p95, max};
    std::fprintf(s.csv, "%lld,%u,%.2f,%.2f,%.2f,%.2f\n", (long long)s.second, frames, mean, p50, p95, max);
    std::fflush(s.csv);
    if (s.log) {
        Log("perf: t=%lld s, %u fps, frame time p50/p95/max %.1f/%.1f/%.1f ms", (long long)s.second, frames, p50, p95,
            max);
    }
}

} // namespace

void OnPresent() {
    auto& s = S();
    const double now = Uptime();
    std::lock_guard lock(s.mu);
    if (!s.csv) {
        s.csv = std::fopen("fps.csv", "w");
        if (!s.csv) Fatal("cannot write fps.csv");
        std::fputs("second,frames,mean_ms,p50_ms,p95_ms,max_ms\n", s.csv);
        const char* l = std::getenv("RT_PERF_LOG");
        s.log = !(l && std::strcmp(l, "0") == 0);
    }
    const s64 second = static_cast<s64>(now);
    if (s.second < 0) s.second = second;
    while (s.second < second) { // close the finished second, and any empty ones after it
        Flush(s);
        s.gaps_ms.clear();
        ++s.second;
    }
    // The first frame has no gap; it starts the clock.
    if (s.last_frame >= 0) s.gaps_ms.push_back((now - s.last_frame) * 1000.0);
    s.last_frame = now;
    s.last_frame_uptime.store(now, std::memory_order_relaxed);
    s.frames.fetch_add(1, std::memory_order_relaxed);
}

Second LastSecond() {
    auto& s = S();
    std::lock_guard lock(s.mu);
    return s.last;
}

u64 Frames() {
    return S().frames.load(std::memory_order_relaxed);
}

double LastFrameUptime() {
    return S().last_frame_uptime.load(std::memory_order_relaxed);
}

} // namespace rt::perf
