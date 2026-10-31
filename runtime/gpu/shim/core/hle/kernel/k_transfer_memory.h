// Shim: the guest range a transfer-memory handle names (audio_core reads the address).
#pragma once
#include "common/common_types.h"
#include "common/typed_address.h"

namespace Kernel {

class KTransferMemory {
public:
    KTransferMemory(VAddr address_, u64 size_) : address{address_}, size{size_} {}
    Common::ProcessAddress GetSourceAddress() const {
        return address;
    }
    u64 GetSize() const {
        return size;
    }

private:
    VAddr address;
    u64 size;
};

} // namespace Kernel
