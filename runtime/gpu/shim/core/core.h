// Shim for drippu's core/core.h: the part of Core::System that video_core
// and audio_core use. It shadows
// the real header through the include order; the methods are implemented by
// the runtime (runtime/gpu/system.cpp).
#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>
#include "common/common_types.h"

namespace Tegra {
class GPU;
namespace Host1x {
class Host1x;
}
} // namespace Tegra

namespace AudioCore {
class AudioCore;
}
namespace Kernel {
class KernelCore;
}

namespace Core {
class DeviceMemory;
class PerfStats;
namespace Timing {
class CoreTiming;
}
namespace Memory {
class Memory;
}

class System {
public:
    [[nodiscard]] static System& GetInstance();

    [[nodiscard]] bool IsPoweredOn() const;
    void GatherGPUDirtyMemory(std::function<void(PAddr, size_t)>& callback);
    [[nodiscard]] Core::Memory::Memory& ApplicationMemory();
    [[nodiscard]] Tegra::GPU& GPU();
    [[nodiscard]] const Tegra::GPU& GPU() const;
    [[nodiscard]] Tegra::Host1x::Host1x& Host1x();
    [[nodiscard]] const Tegra::Host1x::Host1x& Host1x() const;
    [[nodiscard]] Core::DeviceMemory& DeviceMemory();
    [[nodiscard]] const Core::DeviceMemory& DeviceMemory() const;
    [[nodiscard]] Timing::CoreTiming& CoreTiming();
    [[nodiscard]] const Timing::CoreTiming& CoreTiming() const;
    [[nodiscard]] Core::PerfStats& GetPerfStats();
    void RegisterHostThread();

    // audio_core
    [[nodiscard]] AudioCore::AudioCore& AudioCore();
    [[nodiscard]] Kernel::KernelCore& Kernel();
    [[nodiscard]] bool IsShuttingDown() const { return false; }
    [[nodiscard]] bool IsPaused() const { return false; }
};

} // namespace Core
