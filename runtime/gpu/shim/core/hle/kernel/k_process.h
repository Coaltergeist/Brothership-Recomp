// Shim: video_core includes this without using it; audio_core reads and
// writes the process's memory through it.
#pragma once
#include "common/alignment.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h" // the real header provides these transitively; pool_mapper.cpp relies on it

namespace Kernel {

class KProcess {
public:
    Core::Memory::Memory& GetMemory();
    void Open(KernelCore&) {}
    void Close(KernelCore&) {}
};

} // namespace Kernel
