// Shim for drippu's core/memory.h: the CPU-side memory interface that the
// GPU's device memory manager uses (verified: device_memory_manager.inc Map,
// UpdatePagesCachedCount; gpu_device_memory_manager.cpp MarkRegionCaching).
// Implemented by the runtime over its guest address space.
#pragma once
#include <cstddef>
#include <span>
#include "common/common_types.h"
#include "common/typed_address.h"
#include "core/guest_memory.h"

namespace Core::Memory {

constexpr std::size_t YUZU_PAGEBITS = 12;
constexpr u64 YUZU_PAGESIZE = 1ULL << YUZU_PAGEBITS;
constexpr u64 YUZU_PAGEMASK = YUZU_PAGESIZE - 1;

class Memory {
public:
    /// Host pointer for a guest VA, or null if unmapped. No access checks.
    u8* GetPointerSilent(Common::ProcessAddress vaddr);
    /// The GPU caches [vaddr, vaddr + size) (cached) or no longer does: CPU
    /// writes there must reach GPU::OnCPUWrite / the dirty-memory gatherer.
    void RasterizerMarkRegionCached(Common::ProcessAddress vaddr, u64 size, bool cached);

    // Block access, as core/guest_memory.h's GuestMemory uses it. "Unsafe" and
    // "Cached" differ in drippu only in GPU cache flushing (there is none here).
    static constexpr bool HAS_FLUSH_INVALIDATION = false;
    bool ReadBlock(Common::ProcessAddress src, void* dest, std::size_t size);
    bool ReadBlockUnsafe(Common::ProcessAddress src, void* dest, std::size_t size);
    bool WriteBlock(Common::ProcessAddress dest, const void* src, std::size_t size);
    bool WriteBlockUnsafe(Common::ProcessAddress dest, const void* src, std::size_t size);
    bool WriteBlockCached(Common::ProcessAddress dest, const void* src, std::size_t size);
    bool ZeroBlock(Common::ProcessAddress dest, std::size_t size);
    void Write32(Common::ProcessAddress addr, u32 value) {
        WriteBlock(addr, &value, sizeof value);
    }
    const u8* GetSpan(VAddr src, std::size_t size) const;
    u8* GetSpan(VAddr src, std::size_t size);
    void FlushRegion(Common::ProcessAddress, u64) {}
    void InvalidateRegion(Common::ProcessAddress, u64) {}

    /// For drippu's recompiler store guards (shim/core/arm/recomp), never watching here.
    struct PageTableView {
        const void* entries = nullptr;
    };
    PageTableView GetPageTableView() const { return {}; }
};

template <typename T, GuestMemoryFlags FLAGS>
using CpuGuestMemory = GuestMemory<Core::Memory::Memory, T, FLAGS>;
template <typename T, GuestMemoryFlags FLAGS>
using CpuGuestMemoryScoped = GuestMemoryScoped<Core::Memory::Memory, T, FLAGS>;

} // namespace Core::Memory
