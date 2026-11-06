// NVNmemoryPool and NVNbuffer, their builders, and GPU addresses.
//
// A memory pool is game memory (MemoryPoolBuilderSetStorage). Its GPU address
// is that memory's guest address: the game treats GPU addresses as numbers it
// adds offsets to and hands back, so any 1:1 numbering works (LASRecomp does
// the same), and one interval map turns every address a command carries into
// (pool, offset). The data itself stays in guest memory; how the GPU sees it
// (copies at submit, written back on completion) is Phase 5's.
// Virtual pools (MemoryPoolMapVirtual) are not implemented: no game so far
// uses them.
#include "images.h"
#include "nvn.h"

#include "../threads.h"
#include "tap.h"

#include <atomic>
#include <map>
#include <mutex>
#include <shared_mutex>

namespace rt::nvn {

namespace {

// NVNmemoryPoolFlags (verified 2026-10-15, nvn_api_report_verification.md)
constexpr u32 kCpuNoAccess = 0x1, kGpuNoAccess = 0x8, kVirtual = 0x200;

struct MemoryPoolBuilder { // inline in NVNmemoryPoolBuilder
    u64 device;
    u32 flags;
    u32 pad;
    u64 storage, size;
};

struct MemoryPool : Object {
    static constexpr Type kType = Type::kMemoryPool;
    u64 storage = 0, size = 0;
    u32 flags = 0;
};

struct BufferBuilder { // inline in NVNbufferBuilder
    u64 device;
    u64 pool;
    u64 offset;
    u64 size;
};

struct Buffer : Object {
    static constexpr Type kType = Type::kBuffer;
    u64 pool = 0, offset = 0, size = 0, address = 0;
    u64 debug_id = 0;
};

std::atomic<u64> g_debug_ids{0};

// Pools by GPU address (= guest address of their storage).
std::shared_mutex g_pools_mu;
std::map<u64, MemoryPool*> g_pools; // storage -> pool

} // namespace

PoolRange FindPool(u64 gpu_address) {
    std::shared_lock lock(g_pools_mu);
    auto it = g_pools.upper_bound(gpu_address);
    if (it == g_pools.begin()) return {};
    --it;
    const MemoryPool* p = it->second;
    if (gpu_address - p->storage >= p->size) return {};
    return {p->guest, p->storage, p->size, gpu_address - p->storage};
}

PoolRange PoolOf(u64 pool) {
    const MemoryPool* p = Get<MemoryPool>(pool);
    return {p->guest, p->storage, p->size, 0};
}

// --- MemoryPoolBuilder -----------------------------------------------------------------------

NVN_PROC(nvnMemoryPoolBuilderSetDevice, void, u64 builder, u64 device) {
    auto b = Load<MemoryPoolBuilder>(builder);
    b.device = device;
    Store(builder, b);
}
NVN_PROC(nvnMemoryPoolBuilderSetDefaults, void, u64 builder) {
    auto b = Load<MemoryPoolBuilder>(builder);
    b.flags = 0x21; // CPU_NO_ACCESS | GPU_CACHED
    b.storage = 0;
    b.size = 0;
    Store(builder, b);
}
NVN_PROC(nvnMemoryPoolBuilderSetFlags, void, u64 builder, u32 flags) {
    auto b = Load<MemoryPoolBuilder>(builder);
    b.flags = flags;
    Store(builder, b);
}
NVN_PROC(nvnMemoryPoolBuilderSetStorage, void, u64 builder, u64 memory, u64 size) {
    auto b = Load<MemoryPoolBuilder>(builder);
    b.storage = memory;
    b.size = size;
    Store(builder, b);
}
NVN_PROC(nvnMemoryPoolBuilderGetDevice, u64, u64 builder) { return Load<MemoryPoolBuilder>(builder).device; }
NVN_PROC(nvnMemoryPoolBuilderGetFlags, u32, u64 builder) { return Load<MemoryPoolBuilder>(builder).flags; }
NVN_PROC(nvnMemoryPoolBuilderGetMemory, u64, u64 builder) { return Load<MemoryPoolBuilder>(builder).storage; }
NVN_PROC(nvnMemoryPoolBuilderGetSize, u64, u64 builder) { return Load<MemoryPoolBuilder>(builder).size; }

// --- MemoryPool --------------------------------------------------------------------------------

NVN_PROC(nvnMemoryPoolInitialize, bool, u64 pool, u64 builder) {
    const auto b = Load<MemoryPoolBuilder>(builder);
    if (b.flags & kVirtual) Fatal("nvn: virtual memory pools are not implemented yet (flags 0x%x)", b.flags);
    if (!b.storage || (b.storage & 0xFFF) || (b.size & 0xFFF) || !b.size) {
        Log("nvn: nvnMemoryPoolInitialize: storage 0x%llx size 0x%llx not 4 KB aligned: refused",
            (unsigned long long)b.storage, (unsigned long long)b.size);
        return false;
    }
    MemoryPool* p = Create<MemoryPool>(pool);
    p->storage = b.storage;
    p->size = b.size;
    p->flags = b.flags;
    std::unique_lock lock(g_pools_mu);
    auto [it, fresh] = g_pools.emplace(p->storage, p);
    if (!fresh) Fatal("nvn: two memory pools on storage 0x%llx", (unsigned long long)p->storage);
    return true;
}
NVN_PROC(nvnMemoryPoolFinalize, void, u64 pool) {
    MemoryPool* p = Get<MemoryPool>(pool);
    {
        std::unique_lock lock(g_pools_mu);
        g_pools.erase(p->storage);
    }
    DropImagesIn(p->storage, p->storage + p->size); // its memory is gone (images.cpp EvictImages)
    Destroy<MemoryPool>(pool);
}
NVN_PROC(nvnMemoryPoolSetDebugLabel, void, u64 pool, u64 label) { Get<MemoryPool>(pool); }
NVN_PROC(nvnMemoryPoolMap, u64, u64 pool) {
    const MemoryPool* p = Get<MemoryPool>(pool);
    return p->flags & kCpuNoAccess ? 0 : p->storage;
}
NVN_PROC(nvnMemoryPoolGetBufferAddress, u64, u64 pool) {
    const MemoryPool* p = Get<MemoryPool>(pool);
    return p->flags & kGpuNoAccess ? 0 : p->storage;
}
NVN_PROC(nvnMemoryPoolGetFlags, u32, u64 pool) { return Get<MemoryPool>(pool)->flags; }
NVN_PROC(nvnMemoryPoolGetSize, u64, u64 pool) { return Get<MemoryPool>(pool)->size; }
// The GPU reads guest memory as it is at submit and writes back before a sync
// signals (Phase 5): nothing to flush or invalidate.
NVN_PROC(nvnMemoryPoolFlushMappedRange, void, u64 pool, s64 offset, u64 size) { Get<MemoryPool>(pool); }
NVN_PROC(nvnMemoryPoolInvalidateMappedRange, void, u64 pool, s64 offset, u64 size) { Get<MemoryPool>(pool); }

// --- BufferBuilder -----------------------------------------------------------------------------

NVN_PROC(nvnBufferBuilderSetDevice, void, u64 builder, u64 device) {
    auto b = Load<BufferBuilder>(builder);
    b.device = device;
    Store(builder, b);
}
NVN_PROC(nvnBufferBuilderSetDefaults, void, u64 builder) {
    auto b = Load<BufferBuilder>(builder);
    b.pool = b.offset = b.size = 0;
    Store(builder, b);
}
NVN_PROC(nvnBufferBuilderSetStorage, void, u64 builder, u64 pool, s64 offset, u64 size) {
    auto b = Load<BufferBuilder>(builder);
    b.pool = pool;
    b.offset = static_cast<u64>(offset);
    b.size = size;
    Store(builder, b);
}
NVN_PROC(nvnBufferBuilderGetDevice, u64, u64 builder) { return Load<BufferBuilder>(builder).device; }
NVN_PROC(nvnBufferBuilderGetMemoryPool, u64, u64 builder) { return Load<BufferBuilder>(builder).pool; }
NVN_PROC(nvnBufferBuilderGetMemoryOffset, s64, u64 builder) {
    return static_cast<s64>(Load<BufferBuilder>(builder).offset);
}
NVN_PROC(nvnBufferBuilderGetSize, u64, u64 builder) { return Load<BufferBuilder>(builder).size; }

// --- Buffer -------------------------------------------------------------------------------------

NVN_PROC(nvnBufferInitialize, bool, u64 buffer, u64 builder) {
    const auto b = Load<BufferBuilder>(builder);
    const MemoryPool* p = Get<MemoryPool>(b.pool);
    if (b.offset + b.size > p->size || b.offset + b.size < b.offset) {
        Log("nvn: nvnBufferInitialize: 0x%llx+0x%llx outside its pool (0x%llx bytes): refused",
            (unsigned long long)b.offset, (unsigned long long)b.size, (unsigned long long)p->size);
        return false;
    }
    Buffer* buf = Create<Buffer>(buffer);
    buf->pool = b.pool;
    buf->offset = b.offset;
    buf->size = b.size;
    buf->address = p->flags & kGpuNoAccess ? 0 : p->storage + b.offset;
    buf->debug_id = ++g_debug_ids;
    return true;
}
NVN_PROC(nvnBufferFinalize, void, u64 buffer) { Destroy<Buffer>(buffer); }
NVN_PROC(nvnBufferSetDebugLabel, void, u64 buffer, u64 label) { Get<Buffer>(buffer); }
NVN_PROC(nvnBufferMap, u64, u64 buffer) {
    const Buffer* b = Get<Buffer>(buffer);
    const MemoryPool* p = Get<MemoryPool>(b->pool);
    return p->flags & kCpuNoAccess ? 0 : p->storage + b->offset;
}
NVN_PROC(nvnBufferGetAddress, u64, u64 buffer) { return Get<Buffer>(buffer)->address; }
NVN_PROC(nvnBufferGetMemoryPool, u64, u64 buffer) { return Get<Buffer>(buffer)->pool; }
NVN_PROC(nvnBufferGetMemoryOffset, s64, u64 buffer) { return static_cast<s64>(Get<Buffer>(buffer)->offset); }
NVN_PROC(nvnBufferGetSize, u64, u64 buffer) { return Get<Buffer>(buffer)->size; }
NVN_PROC(nvnBufferGetDebugID, u64, u64 buffer) { return Get<Buffer>(buffer)->debug_id; }
NVN_PROC(nvnBufferFlushMappedRange, void, u64 buffer, s64 offset, u64 size) { Get<Buffer>(buffer); }
NVN_PROC(nvnBufferInvalidateMappedRange, void, u64 buffer, s64 offset, u64 size) { Get<Buffer>(buffer); }

// --- Model checks (tap mode, RT_NVN_TAP=1) ------------------------------------------------------
// While the real driver runs: what the game configured for each pool (read back
// with the SDK's own builder getters), and whether every Map, GetFlags,
// GetSize and GetBufferAddress the driver answers agrees with this file's
// rules. Logged at the 1st, 10th, 100th ... check of each function, and each
// difference (up to 10 per function).

namespace {

struct ModelPool {
    u64 storage, size;
    u32 flags;
    u64 driver_address; // the real driver's GPU address, for the log
};
std::mutex g_model_mu;
std::map<u64, ModelPool> g_model_pools; // the game's NVNmemoryPool -> what it was built with
std::map<std::string, std::pair<u64, u64>> g_model_counts; // function -> (checked, different)

void ModelResult(const std::string& fn, bool same, u64 pool, u64 driver, u64 layer) {
    std::lock_guard lock(g_model_mu);
    auto& [checked, different] = g_model_counts[fn];
    ++checked;
    if (!same && ++different <= 10) {
        Log("nvn model: %s(0x%llx): driver 0x%llx, layer would answer 0x%llx", fn.c_str(), (unsigned long long)pool,
            (unsigned long long)driver, (unsigned long long)layer);
    }
    u64 n = checked;
    while (n % 10 == 0) n /= 10;
    if (n == 1) { // 1, 10, 100, ...
        Log("nvn model: %s: %llu calls checked, %llu different", fn.c_str(), (unsigned long long)checked,
            (unsigned long long)different);
    }
}

bool ModelPoolOf(u64 pool, ModelPool& out) {
    std::lock_guard lock(g_model_mu);
    auto it = g_model_pools.find(pool);
    if (it == g_model_pools.end()) return false;
    out = it->second;
    return true;
}

void ShadowPoolInitialize(const std::string&, const GuestContext& before, const GuestContext& after) {
    if (!(after.x[0] & 0xFF)) return;
    GuestContext* c = CurrentContext();
    const u64 pool = before.x[0], builder = before.x[1];
    ModelPool m{};
    m.storage = hle::CallGuest(c, RealProc(c, "nvnMemoryPoolBuilderGetMemory"), {builder}, true);
    m.size = hle::CallGuest(c, RealProc(c, "nvnMemoryPoolBuilderGetSize"), {builder}, true);
    m.flags = static_cast<u32>(hle::CallGuest(c, RealProc(c, "nvnMemoryPoolBuilderGetFlags"), {builder}, true));
    m.driver_address = hle::CallGuest(c, RealProc(c, "nvnMemoryPoolGetBufferAddress"), {pool}, true);
    std::lock_guard lock(g_model_mu);
    if (g_model_pools.size() < 20) {
        Log("nvn model: pool 0x%llx: storage 0x%llx size 0x%llx flags 0x%x, driver GPU address 0x%llx",
            (unsigned long long)pool, (unsigned long long)m.storage, (unsigned long long)m.size, m.flags,
            (unsigned long long)m.driver_address);
    }
    g_model_pools[pool] = m;
}

void ShadowPoolFinalize(const std::string&, const GuestContext& before, const GuestContext&) {
    std::lock_guard lock(g_model_mu);
    g_model_pools.erase(before.x[0]);
}

void ShadowPoolQuery(const std::string& fn, const GuestContext& before, const GuestContext& after) {
    ModelPool m;
    if (!ModelPoolOf(before.x[0], m)) return;
    const u64 driver = after.x[0];
    if (fn == "nvnMemoryPoolMap") {
        const u64 layer = m.flags & kCpuNoAccess ? 0 : m.storage;
        ModelResult(fn, driver == layer, before.x[0], driver, layer);
    } else if (fn == "nvnMemoryPoolGetFlags") {
        ModelResult(fn, static_cast<u32>(driver) == m.flags, before.x[0], driver, m.flags);
    } else if (fn == "nvnMemoryPoolGetSize") {
        ModelResult(fn, driver == m.size, before.x[0], driver, m.size);
    } else if (fn == "nvnMemoryPoolGetBufferAddress") {
        // The numbers differ by design; what must agree: no address exactly
        // when the GPU has no access, and the same address on every call.
        const u64 layer = m.flags & kGpuNoAccess ? 0 : m.storage;
        ModelResult(fn, (driver == 0) == (layer == 0) && driver == m.driver_address, before.x[0], driver, layer);
    }
}

} // namespace

u64 DriverGpuToGuest(u64 gpu) {
    std::lock_guard lock(g_model_mu);
    for (const auto& [pool, m] : g_model_pools) {
        if (m.driver_address && gpu - m.driver_address < m.size) return m.storage + (gpu - m.driver_address);
    }
    return 0;
}

void InstallMemoryShadows() {
    tap::AddShadow("nvnMemoryPoolInitialize", ShadowPoolInitialize);
    tap::AddShadow("nvnMemoryPoolFinalize", ShadowPoolFinalize);
    for (const char* fn : {"nvnMemoryPoolMap", "nvnMemoryPoolGetFlags", "nvnMemoryPoolGetSize",
                           "nvnMemoryPoolGetBufferAddress"}) {
        tap::AddShadow(fn, ShadowPoolQuery);
    }
}

} // namespace rt::nvn
