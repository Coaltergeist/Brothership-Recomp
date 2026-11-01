// Runtime for lifted Switch code: guest memory, dispatch, and kernel HLE.
#pragma once

#include <atomic>
#include <shared_mutex>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include "recomp_runtime.h"
#include "game/game.h"

// Emitted by the lifter (nx_table.c), reached through rt::Game() (game/game.h).
typedef struct RtFunc {
    uint64_t offset;
    void (*fn)(GuestContext*);
} RtFunc;
typedef struct RtSegment {
    uint64_t offset, size;
    uint32_t perm; // 1 read, 2 write, 4 execute
} RtSegment;

// What the lifted code calls (its rt_svc, rt_jump, rt_break, recomp_lookup forward here: game/entry.c, NxHost).
void rt_host_svc(GuestContext* c, uint32_t imm);
void rt_host_jump(GuestContext* c, uint64_t target);
void rt_host_break(GuestContext* c, uint32_t imm, uint64_t pc);
BlockFn rt_host_lookup(uint64_t pc);
}

namespace rt {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using s32 = int32_t;
using u64 = uint64_t;
using s64 = int64_t;

/// The lifted code's tables (game/game.h): linked in (NXRT_GAME=static) or from the loaded module (module). Valid
/// from LoadGame() on (main, before the image is read).
const NxGame& Game();
/// The lifted code: linked in, or the game module - RT_GAME_MODULE, else `module`, else beside the executable. A
/// module whose NxGame.id is not `want_id` (when given: a player start, build.h) stops the run.
void LoadGame(const char* module = nullptr, const char* want_id = nullptr);
bool GameLoaded(); // (the crash report: a crash before LoadGame)

// Guest address space: Horizon's 39-bit layout (NPDM address-space type 3).
// One host reservation covers all of it; only the regions mapped through Memory
// are accessible (an access anywhere else faults: the crash report). Region sizes are Horizon's;
// placement is fixed rather than randomised.
constexpr u64 kAddressSpaceEnd = 1ULL << 39;
constexpr u64 kWindowBase = 0;
constexpr u64 kWindowSize = kAddressSpaceEnd;
// Where the window lies in the host: fixed, so lifted code can reach guest memory as this constant plus the
// guest address (old_plans/FASTMEM_PLAN.md Phase 2). 64 TB: far from where Linux and Windows place images, heaps and
// stacks (1 TB was taken once on Windows: high-entropy ASLR spreads low allocations that far).
constexpr uintptr_t kHostWindowBase = 0x4000'0000'0000;
constexpr u64 kImageBase = 0x0800'0000;
constexpr u64 kAslrRegion = 0x0800'0000, kAslrRegionSize = kAddressSpaceEnd - kAslrRegion;
constexpr u64 kStackRegion = 0x2000'0000, kStackRegionSize = 0x8000'0000;       // 2 GB
constexpr u64 kTlsBase = 0xA000'0000;                                             // kernel map region
// Runtime-owned memory at the top of the kernel map region (TLS grows up from
// kTlsBase): the guest sees it as inaccessible. Holds the overlay (overlay.h).
constexpr u64 kRuntimeRegion = 0xF000'0000, kRuntimeRegionSize = 0x40'0000;
// Fake guest addresses of host functions (hle.h), above the runtime region:
// reserved and inaccessible, reached only as branch targets.
constexpr u64 kHostFnRegion = 0xF100'0000, kHostFnRegionSize = 0x10'0000;
// Guest memory host code hands to guest functions it calls (strings, stand-in
// objects): readable and writable, never used by the game itself (hle.h).
constexpr u64 kHostScratchRegion = 0xF110'0000, kHostScratchRegionSize = 0x10'0000;
constexpr u64 kHeapRegion = 0x1'0000'0000, kHeapRegionSize = 0x2'0000'0000;      // 8 GB
constexpr u64 kAliasRegion = 0x10'0000'0000, kAliasRegionSize = 0x10'0000'0000;  // 64 GB
constexpr u64 kMainStackSize = 0x10'0000;
// Memory an application gets (GetInfo TotalMemorySize) *(assumption: the usual
// application pool; check against an emulator)*, and the title's system
// resource size from its NPDM (META+0x14; Brothership 0x5000000). A non-zero
// system resource size is what enables nn::os virtual address memory (VAMM).
constexpr u64 kTotalMemory = 0xCD50'0000;
constexpr u64 kSystemResourceSize = 0x500'0000;

// Return address the "loader" hands the entry point; jumping here ends the program.
constexpr u64 kLoaderReturn = 0xFFFF'FFFF'FFFF'F000;

enum Perm : u32 { kRead = 1, kWrite = 2, kExec = 4 };

// MemoryInfo.type values (Horizon MemoryState low byte).
enum MemType : u32 {
    kMemUnmapped = 0x00, kMemCodeStatic = 0x03, kMemCodeMutable = 0x04, kMemHeap = 0x05,
    kMemNormal = 0x05, // MapPhysicalMemory: Horizon reports it as the same state as heap
    kMemShared = 0x06,
    kMemStack = 0x0B, kMemThreadLocal = 0x0C, kMemInaccessible = 0x10,
};

struct Region {
    u64 base, size;
    u32 perm;
    u32 type;
    std::string name;
};

class Memory {
public:
    void Init();
    void Map(u64 base, u64 size, u32 perm, u32 type, const std::string& name);
    // As Map, then merges the region with the neighbours it touches that have the same permissions, type and
    // name, as Horizon's memory block manager merges identical blocks. Only for ranges removed by UnmapRange
    // (MapPhysicalMemory): Unmap(base) removes a whole region by its base.
    void MapMerged(u64 base, u64 size, u32 perm, u32 type, const std::string& name);
    void Resize(const std::string& name, u64 size);
    // Changes permissions of [base, base+size), splitting regions as needed.
    bool Protect(u64 base, u64 size, u32 perm);
    void Unmap(u64 base);
    // Removes [base, base+size) from whatever regions it overlaps (splitting them).
    void UnmapRange(u64 base, u64 size);
    const Region* Find(u64 va) const;
    const std::vector<Region>& Regions() const { return regions_; }
    /// The base of the first region that starts above va (kAddressSpaceEnd if there is none).
    u64 NextRegionBase(u64 va) const;
    /// Calls f(region) for each region that overlaps [lo, hi).
    template <typename F>
    void ForEachOverlapping(u64 lo, u64 hi, F f) const {
        std::shared_lock lock(mu_);
        const auto [a, b] = OverlapLocked(lo, hi);
        for (size_t i = a; i < b; ++i) f(regions_[i]);
    }

    // Host pointer for a checked guest access; aborts with a diagnostic on a fault.
    u8* Access(u64 va, u64 size, u32 perm);
    u8* Raw(u64 va) { return host_ + (va - kWindowBase); }

    void Write(u64 va, const void* data, u64 size);
    void Read(u64 va, void* data, u64 size);
    std::string ReadString(u64 va, u64 max);

    // Device-visible memory. The GPU (video_core) addresses guest memory as
    // "physical" offsets into one backing buffer. Pages the GPU can see are
    // moved onto that backing (a memfd) and mapped back at the same guest
    // address, so the CPU's view and pointers stay as they were.
    static constexpr u64 kDeviceBackingSize = 1ULL << 32; // video_core's 4 GB physical window

    /// Makes [va, va + size) device-visible (page granular; pages that already are stay put).
    void MakeDeviceVisible(u64 va, u64 size);
    /// Backing pointer for a device-visible guest address, else null.
    u8* DevicePointer(u64 va) const;
    /// Base of the backing: device physical address 0.
    u8* DeviceBacking();

    // GPU-cached pages. video_core reports the pages its caches hold
    // (RasterizerMarkRegionCached); a checked write to one is passed on to the
    // GPU (gpu::OnCpuWrite), as drippu's Memory does for RasterizerCached pages.
    /// Counts [va, va + size) as cached once more (or once less).
    void MarkGpuCached(u64 va, u64 size, bool cached);
    /// Whether any page of [va, va + size) is GPU-cached.
    bool IsGpuCached(u64 va, u64 size) const;

    // Fastmem: the generated code's inline path (RecompHostMem::page_entries).
    // One entry per 4 KB page: the host pointer of guest address 0 plus a tag in
    // the low bits. Plain mapped pages are resolved inline; GPU-cached pages
    // (tagged) and unmapped ones (0) still take the Load/Store callbacks.
    /// Builds the page table from the current map and hands it to `hm`; the map's
    /// changes keep it current from then on.
    void EnableFastmem(RecompHostMem& hm);

private:
    // A count per 4 KB page, in 256 MB chunks allocated on first use. The GPU
    // thread marks while guest threads test, hence the atomics.
    static constexpr u32 kGpuChunkBits = 28 - 12;
    struct GpuChunk {
        std::atomic<u16> count[1u << kGpuChunkBits];
    };
    std::unique_ptr<std::atomic<GpuChunk*>[]> gpu_cached_ =
        std::make_unique<std::atomic<GpuChunk*>[]>(kAddressSpaceEnd >> 28);
    struct DeviceExtent {
        u64 size, phys;
    };
    std::map<u64, DeviceExtent> device_extents_; // guest va -> backing range
    u8* device_base_ = nullptr;
    u64 device_used_ = 0;

    const Region* FindLocked(u64 va) const;
    /// [first, last) indices of the regions overlapping [lo, hi) (regions are sorted by base and disjoint).
    std::pair<size_t, size_t> OverlapLocked(u64 lo, u64 hi) const;
    void ReleaseDeviceVisibleLocked(u64 va, u64 size);
    /// Recomputes the page-table entries of [lo, hi) from the map (mu_ held alone).
    void RefreshPagesLocked(u64 lo, u64 hi);
    static constexpr uintptr_t kPageTagMask = 3, kPageMemory = 1, kPageDebug = 2, kPageGpuCached = 3;
    std::atomic<uintptr_t>* pages_ = nullptr;

    u8* host_ = nullptr;
    std::vector<Region> regions_;
    // Guest code reads the map from many threads (Access); the kernel changes it
    // (always inside an SVC). Readers share mu_; writers take it alone and bump
    // gen_, which invalidates every thread's cached region.
    mutable std::shared_mutex mu_;
    std::atomic<u64> gen_{1};
};

extern Memory g_mem;
extern bool g_trace_svc;
extern bool g_trace_all; // false: RT_TRACE_SVC=first (rt.cpp)
/// A trace line of an SVC site: every time, or its first 100 times with RT_TRACE_SVC=first.
#define RT_TRACE_LINE(...)                                                                                  \
    do {                                                                                                    \
        if (::rt::g_trace_svc) {                                                                            \
            static std::atomic<unsigned> trace_lines_{0};                                                   \
            if (::rt::g_trace_all || trace_lines_.fetch_add(1, std::memory_order_relaxed) < 100) Log(__VA_ARGS__); \
        }                                                                                                   \
    } while (0)

[[noreturn]] void Fatal(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void Log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string Describe(u64 va); // "0x... (symbol+off)"
/// The lifted function starting at `va`, or null. Through a page index (each 4 KB page of the image: where
/// its functions start in rt_funcs), so a lookup reads one index entry and a few neighbouring functions
/// instead of binary-searching all of them; rt_jump does one per indirect call.
const RtFunc* FindFunc(u64 va);
/// Marks / tells a lifted function as having an override (hle::OverrideGuestFunction), so rt_jump needs to
/// look for one only then.
void MarkOverridden(const RtFunc* f);
bool Overridden(const RtFunc* f);
/// Guest addresses of every symbols.txt entry named `name`.
std::vector<u64> FindSymbol(const std::string& name);
/// Whether a symbols.txt was loaded (a player start loads it when Play is chosen).
bool HaveSymbols();
/// Seconds since the runtime started (for timing logs).
double Uptime();
[[noreturn]] void Exit(int code);
/// The calling thread's guest state for a report (Fatal, a crash): guest thread, pc (as --trace-pc records
/// it), lr, sp.
void LogGuestState();

} // namespace rt
