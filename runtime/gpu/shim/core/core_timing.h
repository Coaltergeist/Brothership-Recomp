// Shim for drippu's core/core_timing.h: what video_core (the GPU tick counter)
// and audio_core (host time, looping events for audio in/out sessions) use.
// Implemented by the runtime (runtime/src/gpu.cpp, runtime/src/audio.cpp).
#pragma once
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include "common/common_types.h"

namespace Core::Timing {

/// A callback for a timed event; a returned duration reschedules it.
using TimedCallback = std::function<std::optional<std::chrono::nanoseconds>(
    s64 time, std::chrono::nanoseconds ns_late)>;

struct EventType {
    explicit EventType(TimedCallback&& callback_, std::string&& name_)
        : callback{std::move(callback_)}, name{std::move(name_)} {}
    TimedCallback callback;
    const std::string name;
    size_t sequence_number = 0;
};

enum class UnscheduleEventType {
    Wait,
    NoWait,
};

class CoreTiming {
public:
    /// GPU ticks (GM20B timer, 614.4 MHz) since startup.
    u64 GetGPUTicks() const;
    std::chrono::nanoseconds GetGlobalTimeNs() const;
    std::chrono::microseconds GetGlobalTimeUs() const;

    /// Runs event_type's callback every resched_time (on a runtime host thread) until unscheduled.
    void ScheduleLoopingEvent(std::chrono::nanoseconds start_time, std::chrono::nanoseconds resched_time,
                              const std::shared_ptr<EventType>& event_type, bool absolute_time = false);
    void UnscheduleEvent(const std::shared_ptr<EventType>& event_type,
                         UnscheduleEventType type = UnscheduleEventType::Wait);
};

std::shared_ptr<EventType> CreateEvent(std::string name, TimedCallback&& callback);

} // namespace Core::Timing
