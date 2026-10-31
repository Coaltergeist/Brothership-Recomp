// Shim for drippu's core/device_memory.h. In drippu this is the emulated DRAM
// (a Common::HostMemory). Here it is the runtime's device memory backing
// (rt::Memory::DeviceBacking, runtime/src/rt.h): guest pages the GPU can see
// live there, and their offset in it is their physical address.
#pragma once
#include <cstdint>
#include "common/common_types.h"

namespace Core {

class DeviceMemory {
public:
    struct Backing {
        u8* base = nullptr;
        u8* BackingBasePointer() const noexcept {
            return base;
        }
    };

    template <typename T>
    PAddr GetRawPhysicalAddr(const T* ptr) const {
        return static_cast<PAddr>(reinterpret_cast<uintptr_t>(ptr) -
                                  reinterpret_cast<uintptr_t>(buffer.base));
    }
    template <typename T>
    T* GetPointerFromRaw(PAddr addr) {
        return reinterpret_cast<T*>(buffer.base + addr);
    }

    Backing buffer;
};

} // namespace Core
