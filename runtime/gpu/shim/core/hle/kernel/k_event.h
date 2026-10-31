// Shim for drippu's KEvent/KReadableEvent as audio_core uses them (Signal,
// Clear, GetReadableEvent). The runtime connects `set` to one of its own
// events (rt::KEvent); it may be called from audio_core's host threads.
#pragma once
#include <functional>
#include "core/hle/kernel/kernel.h"

namespace Kernel {

class KReadableEvent {};

class KEvent {
public:
    void Signal(KernelCore&) {
        if (set) set(true);
    }
    void Clear(KernelCore&) {
        if (set) set(false);
    }
    KReadableEvent& GetReadableEvent() {
        return readable;
    }

    std::function<void(bool signaled)> set;

private:
    KReadableEvent readable;
};

} // namespace Kernel
