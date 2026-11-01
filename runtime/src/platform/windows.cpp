// platform.h on Windows (10 1803 or newer: placeholder memory).
//
// Guest memory. Linux reserves the whole window inaccessible, opens what the
// guest maps (pages appear on first touch) and replaces any 4 KB range by a
// view of the device backing with mmap(MAP_FIXED), atomically. Windows can only
// replace whole allocations, so here every part of the window is one of:
//   - a placeholder (reserved, no access): never mapped, or given back;
//   - a view of a private section: pagefile-backed memory of its own, created
//     when the range is first committed;
//   - a view of the device backing.
// Any 4 KB range can then change kind: its views are unmapped back to
// placeholders, the placeholders split at the range's edges, and the pieces
// mapped again — onto the same section at the same offset (the data stays in
// the section) or onto the device backing. A section lives until no view uses
// it (shared_ptr).
//
// A guest thread touching a range while it is being remapped faults; the
// vectored exception handler below waits for the remap and retries the
// instruction. A touch of memory the guest has not mapped (a placeholder, or a
// kept range, below) is not handled: it reaches the crash report, as on Linux.
//
// Every one of those changes is several system calls, and the guest maps and
// unmaps small ranges all the time (~109,000 MapPhysicalMemory /
// UnmapPhysicalMemory to the title, mostly 4 KB-256 KB). So small ranges stay
// mapped: ZeroWindow up to 4 MB zero-fills the range's private views in place,
// makes them inaccessible (VirtualProtect, per view) and remembers the range as
// "kept"; CommitWindow opens kept ranges again, commits placeholders and leaves
// views that are already there (a kept range costs nothing to map again). Kept
// ranges are released for real once they add up to 1 GB, so pagefile-backed
// memory the guest has given back stays bounded.
#include "platform.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#include <timeapi.h>
#include <tlhelp32.h>
#include <io.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <cwctype>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_COALESCE_PLACEHOLDERS
#define MEM_COALESCE_PLACEHOLDERS 0x00000001
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif

namespace rt::platform {

namespace {

[[noreturn]] void Die(const char* what, uintptr_t a = 0, size_t n = 0) {
    std::fprintf(stderr, "[rt] FATAL: windows memory: %s at 0x%llx+0x%llx (error %lu)\n", what, (unsigned long long)a,
                 (unsigned long long)n, GetLastError());
    std::fflush(stderr);
    std::_Exit(70);
}

// --- The placeholder functions (KernelBase.dll, Windows 10 1803+), loaded at run time as drippu does.
using PFN_VirtualAlloc2 = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
using PFN_MapViewOfFile3 = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*,
                                          ULONG);
using PFN_UnmapViewOfFile2 = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);
PFN_VirtualAlloc2 pVirtualAlloc2;
PFN_MapViewOfFile3 pMapViewOfFile3;
PFN_UnmapViewOfFile2 pUnmapViewOfFile2;

void LoadPlaceholderApi() {
    HMODULE k = GetModuleHandleW(L"kernelbase.dll");
    if (!k) k = LoadLibraryW(L"kernelbase.dll");
    pVirtualAlloc2 = k ? reinterpret_cast<PFN_VirtualAlloc2>(GetProcAddress(k, "VirtualAlloc2")) : nullptr;
    pMapViewOfFile3 = k ? reinterpret_cast<PFN_MapViewOfFile3>(GetProcAddress(k, "MapViewOfFile3")) : nullptr;
    pUnmapViewOfFile2 = k ? reinterpret_cast<PFN_UnmapViewOfFile2>(GetProcAddress(k, "UnmapViewOfFile2")) : nullptr;
    if (!pVirtualAlloc2 || !pMapViewOfFile3 || !pUnmapViewOfFile2) Die("placeholder API missing (Windows 10 1803+)");
}

constexpr uintptr_t kPage = 0x1000;
constexpr uintptr_t kZeroInPlaceMax = 4 << 20; // ZeroWindow up to this size zero-fills instead of releasing
constexpr uintptr_t kKeptMax = 1ULL << 30;     // kept ranges are released when they add up to more

struct Section {
    HANDLE handle;
    explicit Section(HANDLE h) : handle{h} {}
    ~Section() { CloseHandle(handle); }
};

enum class Kind : uint8_t { kHole, kPrivate, kDevice };

struct Seg {
    uintptr_t size;
    Kind kind;
    std::shared_ptr<Section> section; // kPrivate: its section; kDevice: the backing
    uint64_t offset;                  // into the section
};

std::mutex g_mu; // everything below
std::map<uintptr_t, Seg> g_segs; // the window, without gaps
uintptr_t g_lo, g_hi;            // the window
std::shared_ptr<Section> g_device;
std::map<uintptr_t, uintptr_t> g_kept; // start -> end: zeroed by ZeroWindow but still mapped (disjoint, merged)
uintptr_t g_kept_bytes;
std::atomic<DWORD> g_owner{0}; // thread id holding g_mu (the exception handler must not wait for itself)

// RT_MEM_STATS=1: what the paths above cost, printed every second by a thread of their own (so a frame that
// stalls in here still reports). Counts, time spent (including the wait for g_mu) and bytes per path.
struct Stat {
    std::atomic<uint64_t> n{0}, ns{0}, bytes{0};
};
enum StatId { kStatLockWait, kStatFault, kStatCommit, kStatZeroKept, kStatZeroRelease, kStatReleaseKept, kStatDevice,
              kStatSection, kStatCount };
constexpr const char* kStatNames[kStatCount] = {"lock wait", "first touch", "commit", "zero in place", "zero release",
                                                "kept release", "device view", "new section"};
Stat g_stats[kStatCount];

uint64_t NowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

/// Adds the time from construction to destruction (and `bytes`) to one statistic.
struct Timed {
    StatId id;
    uint64_t bytes;
    uint64_t t0 = NowNs();
    ~Timed() {
        Stat& st = g_stats[id];
        st.n.fetch_add(1, std::memory_order_relaxed);
        st.ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
        st.bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
};

void ReportStats() {
    uint64_t last[kStatCount][3] = {};
    const uint64_t start = NowNs();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        char line[1024];
        int len = std::snprintf(line, sizeof line, "[rt] mem: t=%llu s:", (unsigned long long)((NowNs() - start) / 1000000000));
        bool any = false;
        for (int i = 0; i < kStatCount; ++i) {
            const uint64_t n = g_stats[i].n.load(), ns = g_stats[i].ns.load(), b = g_stats[i].bytes.load();
            const uint64_t dn = n - last[i][0], dns = ns - last[i][1], db = b - last[i][2];
            last[i][0] = n, last[i][1] = ns, last[i][2] = b;
            if (!dn || len >= static_cast<int>(sizeof line) - 80) continue;
            any = true;
            len += std::snprintf(line + len, sizeof line - len, " %s %llu (%.1f ms, %.1f MB);", kStatNames[i],
                                 (unsigned long long)dn, dns / 1e6, db / 1048576.0);
        }
        if (!any) continue;
        LockStderr();
        std::fprintf(stderr, "%s kept %.0f MB\n", line, g_kept_bytes / 1048576.0);
        UnlockStderr();
    }
}

struct Locked {
    std::unique_lock<std::mutex> lock{g_mu, std::defer_lock};
    Locked() {
        const uint64_t t0 = NowNs();
        lock.lock();
        Stat& st = g_stats[kStatLockWait];
        st.n.fetch_add(1, std::memory_order_relaxed);
        st.ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
        g_owner.store(GetCurrentThreadId(), std::memory_order_relaxed);
    }
    ~Locked() { g_owner.store(0, std::memory_order_relaxed); }
};

std::shared_ptr<Section> NewSection(uint64_t size) {
    Timed t{kStatSection, size};
    HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                                  static_cast<DWORD>(size >> 32), static_cast<DWORD>(size), nullptr);
    if (!h) Die("CreateFileMapping", 0, size);
    return std::make_shared<Section>(h);
}

void MapOnto(uintptr_t at, const Seg& s) {
    if (!pMapViewOfFile3(s.section->handle, GetCurrentProcess(), reinterpret_cast<void*>(at), s.offset, s.size,
                         MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0)) {
        Die("MapViewOfFile3", at, s.size);
    }
}

/// The segment starting at `at` (there must be one).
std::map<uintptr_t, Seg>::iterator SegAt(uintptr_t at) {
    auto it = g_segs.find(at);
    if (it == g_segs.end()) Die("no segment boundary", at);
    return it;
}

void SetKeptAccess(uintptr_t lo, uintptr_t hi, DWORD prot);

/// Makes `at` a segment boundary (no-op at the window's ends or an existing boundary).
void SplitAt(uintptr_t at) {
    if (at <= g_lo || at >= g_hi) return;
    auto it = g_segs.upper_bound(at);
    --it;
    if (it->first == at) return;
    const uintptr_t start = it->first;
    Seg left = it->second;
    const uintptr_t lsize = at - start;
    Seg right{left.size - lsize, left.kind, left.section, left.offset + lsize};
    left.size = lsize;
    if (left.kind != Kind::kHole && !pUnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(start),
                                                      MEM_PRESERVE_PLACEHOLDER)) {
        Die("UnmapViewOfFile2 (split)", start, left.size + right.size);
    }
    if (!VirtualFree(reinterpret_cast<void*>(start), lsize, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
        Die("placeholder split", start, lsize);
    }
    if (left.kind != Kind::kHole) {
        MapOnto(start, left);
        MapOnto(at, right);
    }
    it->second = left;
    g_segs.emplace(at, right);
    if (left.kind != Kind::kHole) SetKeptAccess(start, at + right.size, PAGE_NOACCESS); // new views are read-write
}

/// [lo, hi) (boundaries already) back to one placeholder segment.
void MakeHole(uintptr_t lo, uintptr_t hi) {
    for (auto it = SegAt(lo); it != g_segs.end() && it->first < hi; ++it) {
        if (it->second.kind != Kind::kHole) {
            if (!pUnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(it->first), MEM_PRESERVE_PLACEHOLDER)) {
                Die("UnmapViewOfFile2", it->first, it->second.size);
            }
            it->second = {it->second.size, Kind::kHole, nullptr, 0};
        }
    }
    auto first = SegAt(lo);
    if (std::next(first) != g_segs.end() && std::next(first)->first < hi) {
        if (!VirtualFree(reinterpret_cast<void*>(lo), hi - lo, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
            Die("placeholder coalesce", lo, hi - lo);
        }
        g_segs.erase(std::next(first), g_segs.lower_bound(hi));
        first->second.size = hi - lo;
    }
}

/// Every placeholder in [lo, hi) (boundaries already) becomes a view of a new private section.
void CommitHoles(uintptr_t lo, uintptr_t hi) {
    for (auto it = SegAt(lo); it != g_segs.end() && it->first < hi; ++it) {
        if (it->second.kind != Kind::kHole) continue;
        it->second = {it->second.size, Kind::kPrivate, NewSection(it->second.size), 0};
        MapOnto(it->first, it->second);
    }
}

/// The segment containing `a` (inside the window).
std::map<uintptr_t, Seg>::iterator SegContaining(uintptr_t a) {
    return std::prev(g_segs.upper_bound(a));
}

/// Commits the placeholders in [lo, hi) and leaves the views already there as they are.
void CommitRange(uintptr_t lo, uintptr_t hi) {
    struct Piece { uintptr_t lo, hi; };
    Piece holes[16];
    size_t n = 0;
    for (auto it = SegContaining(lo); it != g_segs.end() && it->first < hi; ++it) {
        if (it->second.kind != Kind::kHole) continue;
        if (n == std::size(holes)) { // many pieces: commit these, then look again
            break;
        }
        holes[n++] = {std::max(lo, it->first), std::min(hi, it->first + it->second.size)};
    }
    for (size_t i = 0; i < n; ++i) {
        SplitAt(holes[i].lo);
        SplitAt(holes[i].hi);
        CommitHoles(holes[i].lo, holes[i].hi);
    }
    if (n == std::size(holes)) CommitRange(holes[n - 1].hi, hi);
}

/// Zero-fills the private views in [lo, hi); false (and nothing done) if a device view is there.
bool ZeroInPlace(uintptr_t lo, uintptr_t hi) {
    const auto first = SegContaining(lo);
    for (auto it = first; it != g_segs.end() && it->first < hi; ++it) {
        if (it->second.kind == Kind::kDevice) return false;
    }
    for (auto it = first; it != g_segs.end() && it->first < hi; ++it) {
        if (it->second.kind != Kind::kPrivate) continue; // placeholders read as zero already
        const uintptr_t a = std::max(lo, it->first), b = std::min(hi, it->first + it->second.size);
        std::memset(reinterpret_cast<void*>(a), 0, b - a);
    }
    return true;
}

/// Forgets [lo, hi) as kept (it is being mapped, remapped or released).
void KeptRemove(uintptr_t lo, uintptr_t hi) {
    auto it = g_kept.upper_bound(lo);
    if (it != g_kept.begin()) --it;
    while (it != g_kept.end() && it->first < hi) {
        const uintptr_t s = it->first, e = it->second;
        if (e <= lo) {
            ++it;
            continue;
        }
        it = g_kept.erase(it);
        g_kept_bytes -= e - s;
        if (s < lo) {
            g_kept.emplace(s, lo);
            g_kept_bytes += lo - s;
        }
        if (e > hi) {
            g_kept.emplace(hi, e);
            g_kept_bytes += e - hi;
        }
    }
}

void KeptAdd(uintptr_t lo, uintptr_t hi) {
    KeptRemove(lo, hi);
    g_kept_bytes += hi - lo;
    auto it = g_kept.emplace(lo, hi).first;
    if (it != g_kept.begin() && std::prev(it)->second == lo) { // merge with the range before
        std::prev(it)->second = hi;
        it = std::prev(g_kept.erase(it));
    }
    if (auto next = std::next(it); next != g_kept.end() && next->first == it->second) { // and the one after
        it->second = next->second;
        g_kept.erase(next);
    }
}

/// Sets the protection of the views in [lo, hi) (VirtualProtect works within one view at a time).
void SetAccess(uintptr_t lo, uintptr_t hi, DWORD prot) {
    for (auto it = SegContaining(lo); it != g_segs.end() && it->first < hi; ++it) {
        if (it->second.kind == Kind::kHole) continue;
        const uintptr_t a = std::max(lo, it->first), b = std::min(hi, it->first + it->second.size);
        DWORD old;
        if (!VirtualProtect(reinterpret_cast<void*>(a), b - a, prot, &old)) Die("VirtualProtect", a, b - a);
    }
}

/// Sets the protection of the kept parts of [lo, hi).
void SetKeptAccess(uintptr_t lo, uintptr_t hi, DWORD prot) {
    auto it = g_kept.upper_bound(lo);
    if (it != g_kept.begin()) --it;
    for (; it != g_kept.end() && it->first < hi; ++it) {
        const uintptr_t a = std::max(lo, it->first), b = std::min(hi, it->second);
        if (a < b) SetAccess(a, b, prot);
    }
}

/// Whether `a` lies in a kept range.
bool IsKept(uintptr_t a) {
    auto it = g_kept.upper_bound(a);
    return it != g_kept.begin() && std::prev(it)->second > a;
}

/// Releases every kept range: back to placeholders, as a large ZeroWindow does.
void ReleaseKept() {
    Timed t{kStatReleaseKept, g_kept_bytes};
    for (const auto& [s, e] : g_kept) {
        SplitAt(s);
        SplitAt(e);
        MakeHole(s, e);
    }
    g_kept.clear();
    g_kept_bytes = 0;
}

uintptr_t PageDown(uintptr_t a) { return a & ~(kPage - 1); }
uintptr_t PageUp(uintptr_t a) { return (a + kPage - 1) & ~(kPage - 1); }

std::atomic<bool (*)(uintptr_t)> g_write_hook{nullptr}; // writetrack: a write to a page it protected

/// Remaps guest memory while other threads may touch it: see the file's comment. Asks write tracking first
/// (before the remap lock: its faults are frequent and need no remap).
LONG CALLBACK OnFault(EXCEPTION_POINTERS* e) {
    const EXCEPTION_RECORD& r = *e->ExceptionRecord;
    if (r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r.NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    const uintptr_t a = static_cast<uintptr_t>(r.ExceptionInformation[1]);
    if (a < g_lo || a >= g_hi) return EXCEPTION_CONTINUE_SEARCH;
    if (r.ExceptionInformation[0] == 1) { // a write
        if (auto* hook = g_write_hook.load(std::memory_order_acquire); hook && hook(a)) return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (g_owner.load(std::memory_order_relaxed) == GetCurrentThreadId()) return EXCEPTION_CONTINUE_SEARCH;
    // A retry that faults again and again is not a remap race: report it rather than loop.
    thread_local uintptr_t last = 0;
    thread_local unsigned repeats = 0;
    repeats = a == last ? repeats + 1 : 0;
    last = a;
    if (repeats > 1000) return EXCEPTION_CONTINUE_SEARCH;
    Timed t{kStatFault, 0};
    Locked l; // waits for a remap in progress
    if (IsKept(a)) return EXCEPTION_CONTINUE_SEARCH; // unmapped by the guest: the crash report
    auto it = g_segs.upper_bound(a);
    --it;
    if (it->second.kind == Kind::kHole) return EXCEPTION_CONTINUE_SEARCH; // never mapped, or given back
    return EXCEPTION_CONTINUE_EXECUTION; // remapped meanwhile: retry
}

} // namespace

void InitProcess() {
    timeBeginPeriod(1);
    PROCESS_POWER_THROTTLING_STATE pt{};
    pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    pt.StateMask = 0; // (opted out of both)
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof pt);
    SetConsoleOutputCP(CP_UTF8);
}

// --- Guest memory ------------------------------------------------------------

uint8_t* ReserveWindow(size_t size, uintptr_t at) {
    LoadPlaceholderApi();
    if (const char* e = std::getenv("RT_MEM_STATS"); e && *e && *e != '0') {
        std::thread(ReportStats).detach();
    }
    void* p = pVirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(at), size,
                             MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    if (!p) return nullptr;
    Locked l;
    g_lo = reinterpret_cast<uintptr_t>(p);
    g_hi = g_lo + size;
    g_segs.emplace(g_lo, Seg{size, Kind::kHole, nullptr, 0});
    AddVectoredExceptionHandler(1, OnFault);
    return static_cast<uint8_t*>(p);
}

void CommitWindow(uint8_t* p, size_t size) {
    if (!size) return;
    const uintptr_t lo = PageDown(reinterpret_cast<uintptr_t>(p)), hi = PageUp(reinterpret_cast<uintptr_t>(p) + size);
    Timed t{kStatCommit, hi - lo};
    Locked l;
    SetKeptAccess(lo, hi, PAGE_READWRITE);
    KeptRemove(lo, hi);
    CommitRange(lo, hi);
}

bool ProtectReadOnly(uint8_t* p, size_t size) {
    DWORD old;
    return VirtualProtect(p, size, PAGE_READONLY, &old) != 0;
}

bool ProtectReadWrite(uint8_t* p, size_t size) {
    DWORD old;
    return VirtualProtect(p, size, PAGE_READWRITE, &old) != 0;
}

void SetWriteFaultHook(bool (*hook)(uintptr_t a)) { g_write_hook.store(hook, std::memory_order_release); }

void ZeroWindow(uint8_t* p, size_t size) {
    if (!size) return;
    const uintptr_t lo = PageDown(reinterpret_cast<uintptr_t>(p)), hi = PageUp(reinterpret_cast<uintptr_t>(p) + size);
    Timed t{kStatZeroRelease, hi - lo};
    Locked l;
    SetKeptAccess(lo, hi, PAGE_READWRITE); // (ZeroInPlace writes there; the range is closed again below)
    if (hi - lo <= kZeroInPlaceMax && ZeroInPlace(lo, hi)) {
        t.id = kStatZeroKept;
        KeptAdd(lo, hi);
        SetAccess(lo, hi, PAGE_NOACCESS);
        if (g_kept_bytes > kKeptMax) ReleaseKept();
        return;
    }
    KeptRemove(lo, hi);
    SplitAt(lo);
    SplitAt(hi);
    MakeHole(lo, hi); // reads as zero again: the next touch (or commit) gets a fresh section
}

uint8_t* CreateDeviceBacking(size_t size) {
    {
        Locked l;
        g_device = NewSection(size);
    }
    void* p = MapViewOfFile(g_device->handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
    return static_cast<uint8_t*>(p);
}

bool MapDeviceView(uint8_t* p, size_t size, uint64_t offset) {
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p), hi = lo + size;
    if ((lo | size | offset) & (kPage - 1)) return false;
    Timed t{kStatDevice, size};
    Locked l;
    KeptRemove(lo, hi);
    SplitAt(lo);
    SplitAt(hi);
    MakeHole(lo, hi);
    auto it = SegAt(lo);
    it->second = {size, Kind::kDevice, g_device, offset};
    MapOnto(lo, it->second);
    return true;
}

void UnmapDeviceView(uint8_t* p, size_t size) {
    const uintptr_t lo = PageDown(reinterpret_cast<uintptr_t>(p)), hi = PageUp(reinterpret_cast<uintptr_t>(p) + size);
    Locked l;
    SplitAt(lo);
    SplitAt(hi);
    MakeHole(lo, hi);
    CommitHoles(lo, hi); // private zero pages again
}

void* AllocateSparseZeroed(size_t size) {
    // Committed (charged) but not backed until touched; reads as zero.
    return VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

// --- Host files ----------------------------------------------------------------

namespace {
HANDLE H(intptr_t h) { return reinterpret_cast<HANDLE>(h); }
} // namespace

File& File::operator=(File&& other) noexcept {
    if (this != &other) {
        Close();
        handle_ = other.handle_;
        other.handle_ = -1;
    }
    return *this;
}

bool File::Open(const std::filesystem::path& path, Mode mode) {
    Close();
    const DWORD access = mode == Mode::kRead ? GENERIC_READ : mode == Mode::kReadWrite ? GENERIC_READ | GENERIC_WRITE
                                                                                      : GENERIC_WRITE;
    const DWORD disposition = mode == Mode::kCreate ? OPEN_ALWAYS : OPEN_EXISTING;
    HANDLE h = CreateFileW(path.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    handle_ = reinterpret_cast<intptr_t>(h);
    return true;
}

void File::Close() {
    if (handle_ != -1) CloseHandle(H(handle_));
    handle_ = -1;
}

int64_t File::ReadAt(void* out, size_t size, uint64_t offset) const {
    size_t done = 0;
    while (done < size) {
        const uint64_t at = offset + done;
        OVERLAPPED o{};
        o.Offset = static_cast<DWORD>(at);
        o.OffsetHigh = static_cast<DWORD>(at >> 32);
        const DWORD want = static_cast<DWORD>(std::min<size_t>(size - done, 1u << 30));
        DWORD got = 0;
        if (!ReadFile(H(handle_), static_cast<char*>(out) + done, want, &got, &o)) {
            if (GetLastError() == ERROR_HANDLE_EOF) break;
            return -1;
        }
        if (got == 0) break; // end of file
        done += got;
    }
    return static_cast<int64_t>(done);
}

int64_t File::WriteAt(const void* data, size_t size, uint64_t offset) {
    size_t done = 0;
    while (done < size) {
        const uint64_t at = offset + done;
        OVERLAPPED o{};
        o.Offset = static_cast<DWORD>(at);
        o.OffsetHigh = static_cast<DWORD>(at >> 32);
        const DWORD want = static_cast<DWORD>(std::min<size_t>(size - done, 1u << 30));
        DWORD put = 0;
        if (!WriteFile(H(handle_), static_cast<const char*>(data) + done, want, &put, &o) || put == 0) return -1;
        done += put;
    }
    return static_cast<int64_t>(done);
}

bool File::Flush() {
    return FlushFileBuffers(H(handle_)) != 0;
}

bool File::SetSize(uint64_t size) {
    FILE_END_OF_FILE_INFO info{};
    info.EndOfFile.QuadPart = static_cast<LONGLONG>(size);
    return SetFileInformationByHandle(H(handle_), FileEndOfFileInfo, &info, sizeof info) != 0;
}

int64_t File::Size() const {
    LARGE_INTEGER s{};
    return GetFileSizeEx(H(handle_), &s) ? s.QuadPart : -1;
}

// --- Threads and diagnostics -------------------------------------------------

void SleepUntil(std::chrono::steady_clock::time_point deadline) {
    using Clock = std::chrono::steady_clock;
    // The high-resolution timer still wakes ~0.4 ms late on average (measured, RTX host): aim early by how late
    // it has been on this thread (a running average, at most 2 ms), then spin for the rest.
    struct Timer {
        HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        int64_t late_ns = 0;
        ~Timer() {
            if (h) CloseHandle(h);
        }
    };
    thread_local Timer timer;
    if (!timer.h) {
        std::this_thread::sleep_until(deadline);
        return;
    }
    const auto aim = deadline - std::chrono::nanoseconds(timer.late_ns);
    for (;;) {
        const int64_t left = std::chrono::duration_cast<std::chrono::nanoseconds>(aim - Clock::now()).count();
        if (left <= 0) break;
        LARGE_INTEGER due;
        due.QuadPart = -std::max<int64_t>(1, left / 100); // relative, in 100 ns units
        if (!SetWaitableTimerEx(timer.h, &due, 0, nullptr, nullptr, nullptr, 0)) {
            std::this_thread::sleep_until(deadline);
            return;
        }
        WaitForSingleObject(timer.h, INFINITE);
        const int64_t late = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - aim).count();
        if (late >= 0) { // woke at or after the aim: learn from it (an early wake just waits again)
            timer.late_ns = std::min<int64_t>(2'000'000, timer.late_ns + (late - timer.late_ns) / 8);
            break;
        }
    }
    while (Clock::now() < deadline) YieldProcessor();
}

void AttachOutput(const std::string& log_file, bool continuing) {
    const HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
    DWORD mode;
    const DWORD type = err && err != INVALID_HANDLE_VALUE ? GetFileType(err) : FILE_TYPE_UNKNOWN;
    // (a console or a file / pipe: output goes there. The NUL device - a program started in the background, e.g. the
    // start screen's restart into the game - is a character device that is no console: no output, so the file.)
    if (type != FILE_TYPE_UNKNOWN && (type != FILE_TYPE_CHAR || GetConsoleMode(err, &mode))) return;
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        std::freopen("CONOUT$", "w", stdout);
        std::freopen("CONOUT$", "w", stderr);
        return;
    }
    std::error_code ec;
    const std::filesystem::path log(log_file), old = std::filesystem::path(log).replace_extension(".old.log");
    std::filesystem::create_directories(log.parent_path(), ec);
    if (!continuing) std::filesystem::rename(log, old, ec);
    else ec = std::make_error_code(std::errc::file_exists); // (added to, as below)
    // (still open in the run that started this one - its restart into the game -: added to, not emptied)
    if (std::FILE* f = _wfreopen(log.wstring().c_str(), ec ? L"a" : L"w", stderr)) {
        _dup2(_fileno(f), _fileno(stdout)); // (stdout too: --build's progress, the usage line)
    }
}

std::string ExecutablePath() {
    std::wstring w(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, w.data(), static_cast<DWORD>(w.size()));
        if (n < w.size()) {
            w.resize(n);
            break;
        }
        w.resize(w.size() * 2);
    }
    const int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), len, nullptr, nullptr);
    return s;
}

struct Child {
    HANDLE process;
};

namespace {
std::wstring Wide(const std::string& s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

/// One argument as CommandLineToArgvW reads it back (quotes and the backslashes before them escaped).
void AppendQuoted(std::wstring& cmd, const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        cmd += a;
        return;
    }
    cmd += L'"';
    for (size_t i = 0;; ++i) {
        size_t slashes = 0;
        while (i < a.size() && a[i] == L'\\') ++i, ++slashes;
        if (i == a.size()) {
            cmd.append(slashes * 2, L'\\');
            break;
        }
        if (a[i] == L'"') cmd.append(slashes * 2 + 1, L'\\');
        else cmd.append(slashes, L'\\');
        cmd += a[i];
    }
    cmd += L'"';
}
} // namespace

Child* StartChild(const std::vector<std::string>& args, const std::string& log, std::string* error,
                  const std::vector<std::string>& env) {
    // the environment block: this process's, with `env`'s replacing those of the same name (names compare
    // without case, as Windows does)
    std::wstring block;
    if (!env.empty()) {
        std::vector<std::wstring> extra;
        for (const std::string& e : env) extra.push_back(Wide(e));
        const auto name = [](const std::wstring& v) {
            std::wstring n = v.substr(0, v.find(L'=', 1)); // (a leading '=' is part of a drive variable's name)
            for (wchar_t& c : n) c = static_cast<wchar_t>(towupper(c));
            return n;
        };
        if (wchar_t* all = GetEnvironmentStringsW()) {
            for (const wchar_t* p = all; *p; p += wcslen(p) + 1) {
                const std::wstring v = p;
                bool replaced = false;
                for (const std::wstring& x : extra) replaced |= name(x) == name(v);
                if (!replaced) block += v + L'\0';
            }
            FreeEnvironmentStringsW(all);
        }
        for (const std::wstring& x : extra) block += x + L'\0';
        block += L'\0';
    }
    std::wstring cmd;
    for (const std::string& a : args) {
        if (!cmd.empty()) cmd += L' ';
        AppendQuoted(cmd, Wide(a));
    }
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    const HANDLE out = CreateFileW(Wide(log).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE, &sa,
                                   CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    const HANDLE in = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    const auto close = [&] {
        if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
        if (in != INVALID_HANDLE_VALUE) CloseHandle(in);
    };
    if (out == INVALID_HANDLE_VALUE || in == INVALID_HANDLE_VALUE) {
        *error = "cannot open " + log + " (error " + std::to_string(GetLastError()) + ")";
        close();
        return nullptr;
    }
    // only these two handles inherited (other jobs' logs would stay open in every child otherwise)
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    std::vector<unsigned char> list_mem(size);
    auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list_mem.data());
    HANDLE inherit[2] = {out, in};
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof si;
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = in;
    si.StartupInfo.hStdOutput = out;
    si.StartupInfo.hStdError = out;
    const bool listed = InitializeProcThreadAttributeList(list, 1, 0, &size) &&
                        UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, sizeof inherit,
                                                  nullptr, nullptr);
    si.lpAttributeList = listed ? list : nullptr;
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT |
                                       (listed ? EXTENDED_STARTUPINFO_PRESENT : 0),
                                   block.empty() ? nullptr : block.data(), nullptr, &si.StartupInfo, &pi);
    const DWORD err = GetLastError();
    if (listed) DeleteProcThreadAttributeList(list);
    close();
    if (!ok) {
        char text[256] = "";
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, err, 0, text, sizeof text,
                       nullptr);
        *error = std::string(text[0] ? text : "CreateProcess failed") + " (error " + std::to_string(err) + ")";
        while (!error->empty() && (error->back() == '\n' || error->back() == '\r' || error->back() == ' ')) error->pop_back();
        return nullptr;
    }
    CloseHandle(pi.hThread);
    return new Child{pi.hProcess};
}

bool ChildDone(Child* c, int* code) {
    if (WaitForSingleObject(c->process, 0) != WAIT_OBJECT_0) return false;
    DWORD exit = 0;
    GetExitCodeProcess(c->process, &exit);
    *code = static_cast<int>(exit);
    return true;
}

void KillChild(Child* c) {
    TerminateProcess(c->process, 1);
    WaitForSingleObject(c->process, INFINITE);
}

void FreeChild(Child* c) {
    CloseHandle(c->process);
    delete c;
}

uint64_t FreeMemory() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? m.ullAvailPhys : 0;
}

uint64_t CurrentThreadId() {
    return GetCurrentThreadId();
}

void SetCurrentThreadName(const char* name) {
    using PFN_SetThreadDescription = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static const auto set = reinterpret_cast<PFN_SetThreadDescription>(
        GetProcAddress(GetModuleHandleW(L"kernelbase.dll"), "SetThreadDescription")); // Windows 10 1607+
    if (!set) return;
    wchar_t w[64];
    if (MultiByteToWideChar(CP_UTF8, 0, name, -1, w, 64) == 0) return;
    w[63] = 0;
    set(GetCurrentThread(), w);
}

namespace {

void (*g_dump_request)() = nullptr;

BOOL WINAPI OnConsoleEvent(DWORD type) {
    if (type != CTRL_BREAK_EVENT || !g_dump_request) return FALSE; // Ctrl+C keeps ending the run
    g_dump_request();
    return TRUE;
}

/// "module+0xoffset" for a code address (resolve with llvm-symbolizer and the module's .pdb or map).
std::string FrameName(uintptr_t pc) {
    HMODULE m = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(pc), &m)) {
        char path[MAX_PATH];
        if (GetModuleFileNameA(m, path, MAX_PATH)) {
            const char* base = std::strrchr(path, '\\');
            std::snprintf(name, sizeof name, "%s", base ? base + 1 : path);
        }
    }
    char b[MAX_PATH + 32];
    std::snprintf(b, sizeof b, "%s+0x%llx", name, (unsigned long long)(pc - reinterpret_cast<uintptr_t>(m)));
    return b;
}

void PrintFrame(uintptr_t pc) {
    std::fprintf(stderr, "[rt]   %s\n", FrameName(pc).c_str());
}

/// Walks a stopped thread's stack with the unwind tables (no symbols needed).
void WalkStack(CONTEXT ctx) {
    for (int i = 0; i < 48 && ctx.Rip; ++i) {
        PrintFrame(ctx.Rip);
        DWORD64 image = 0;
        PRUNTIME_FUNCTION f = RtlLookupFunctionEntry(ctx.Rip, &image, nullptr);
        if (!f) { // a leaf function: the return address is on top of the stack
            ctx.Rip = *reinterpret_cast<DWORD64*>(ctx.Rsp);
            ctx.Rsp += 8;
            continue;
        }
        void* handler_data = nullptr;
        DWORD64 frame = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, ctx.Rip, f, &ctx, &handler_data, &frame, nullptr);
    }
}

// --- Crash report -------------------------------------------------------------------------------------------

void (*g_describe)() = nullptr;
std::atomic<bool> g_crashing{false};

const char* ExceptionName(DWORD code) {
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION: return "access violation";
    case EXCEPTION_ILLEGAL_INSTRUCTION: return "illegal instruction";
    case EXCEPTION_PRIV_INSTRUCTION: return "privileged instruction";
    case EXCEPTION_STACK_OVERFLOW: return "stack overflow";
    case EXCEPTION_INT_DIVIDE_BY_ZERO: return "integer division by zero";
    case EXCEPTION_IN_PAGE_ERROR: return "in-page error";
    case EXCEPTION_DATATYPE_MISALIGNMENT: return "misaligned access";
    case EXCEPTION_BREAKPOINT: return "breakpoint";
    case 0xE06D7363: return "C++ exception nobody caught";
    default: return "exception";
    }
}

/// nxrt_crash.dmp in the working directory (stacks, threads, modules; not the guest window): open it in WinDbg or
/// Visual Studio next to nxrt.exe. dbghelp is loaded only now, so a normal run does not need it.
void WriteMinidump(EXCEPTION_POINTERS* ep) {
    using PFN_MiniDumpWriteDump = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE, PMINIDUMP_EXCEPTION_INFORMATION,
                                                PMINIDUMP_USER_STREAM_INFORMATION, PMINIDUMP_CALLBACK_INFORMATION);
    HMODULE dbghelp = LoadLibraryW(L"dbghelp.dll");
    const auto write = dbghelp ? reinterpret_cast<PFN_MiniDumpWriteDump>(GetProcAddress(dbghelp, "MiniDumpWriteDump"))
                               : nullptr;
    if (!write) return;
    HANDLE f = CreateFileW(L"nxrt_crash.dmp", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return;
    MINIDUMP_EXCEPTION_INFORMATION info{GetCurrentThreadId(), ep, FALSE};
    const BOOL ok = write(GetCurrentProcess(), GetCurrentProcessId(), f,
                          static_cast<MINIDUMP_TYPE>(MiniDumpWithThreadInfo | MiniDumpWithIndirectlyReferencedMemory),
                          ep ? &info : nullptr, nullptr, nullptr);
    CloseHandle(f);
    std::fprintf(stderr, "[rt]   %s\n", ok ? "minidump: nxrt_crash.dmp" : "minidump: could not be written");
}

/// Everything a crash report says after its first line; then the process ends (exit code 71).
[[noreturn]] void FinishCrash(const CONTEXT& ctx, EXCEPTION_POINTERS* ep) {
    std::fprintf(stderr, "[rt]   host tid %lu, stack:\n", GetCurrentThreadId());
    WalkStack(ctx);
    if (g_describe) g_describe();
    WriteMinidump(ep);
    std::fflush(stderr);
    TerminateProcess(GetCurrentProcess(), 71);
    std::_Exit(71);
}

/// A second crash while reporting one: end at once.
bool FirstCrash() {
    if (!g_crashing.exchange(true)) return true;
    TerminateProcess(GetCurrentProcess(), 71);
    return false;
}

LONG WINAPI OnUnhandledException(EXCEPTION_POINTERS* ep) {
    if (!FirstCrash()) return EXCEPTION_EXECUTE_HANDLER;
    const EXCEPTION_RECORD& r = *ep->ExceptionRecord;
    std::fprintf(stderr, "[rt] CRASH: %s (0x%08lx) at %s\n", ExceptionName(r.ExceptionCode), r.ExceptionCode,
                 FrameName(reinterpret_cast<uintptr_t>(r.ExceptionAddress)).c_str());
    if ((r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) &&
        r.NumberParameters >= 2) {
        const ULONG_PTR kind = r.ExceptionInformation[0];
        const uintptr_t a = static_cast<uintptr_t>(r.ExceptionInformation[1]);
        std::fprintf(stderr, "[rt]   %s address 0x%llx\n", kind == 0 ? "reading" : kind == 1 ? "writing" : "executing",
                     (unsigned long long)a);
        if (a >= g_lo && a < g_hi) {
            std::fprintf(stderr, "[rt]   guest address 0x%llx: not mapped by the guest\n", (unsigned long long)(a - g_lo));
        }
    }
    FinishCrash(*ep->ContextRecord, ep);
}

[[noreturn]] void CrashHere(const char* what) {
    if (!FirstCrash()) std::_Exit(71);
    std::fprintf(stderr, "[rt] CRASH: %s\n", what);
    CONTEXT ctx{};
    RtlCaptureContext(&ctx);
    FinishCrash(ctx, nullptr);
}

void OnTerminate() {
    CrashHere("std::terminate (an exception nobody caught, or one thrown through a function that may not throw)");
}

void OnAbort(int) {
    CrashHere("abort()");
}

} // namespace

void InstallCrashReport(void (*describe)()) {
    g_describe = describe;
    // No Windows error-report dialog or C runtime abort box: the log has the report.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    SetUnhandledExceptionFilter(OnUnhandledException);
    std::set_terminate(OnTerminate);
    std::signal(SIGABRT, OnAbort);
}

void InstallDumpTrigger(void (*request)()) {
    g_dump_request = request;
    SetConsoleCtrlHandler(OnConsoleEvent, TRUE);
}

void DumpHostStacks() {
    const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te{};
    te.dwSize = sizeof te;
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
        HANDLE t = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
                              te.th32ThreadID);
        if (!t) continue;
        if (SuspendThread(t) != static_cast<DWORD>(-1)) {
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_FULL;
            LockStderr();
            std::fprintf(stderr, "[rt] --- host tid %lu\n", te.th32ThreadID);
            if (GetThreadContext(t, &ctx)) WalkStack(ctx);
            UnlockStderr();
            ResumeThread(t);
        }
        CloseHandle(t);
    }
    CloseHandle(snap);
    CONTEXT ctx{};
    RtlCaptureContext(&ctx);
    LockStderr();
    std::fprintf(stderr, "[rt] --- host tid %lu (this thread)\n", self);
    WalkStack(ctx);
    UnlockStderr();
}

void LockStderr() {
    _lock_file(stderr);
}

void UnlockStderr() {
    _unlock_file(stderr);
}

} // namespace rt::platform

// --- Allocation statistics (RT_ALLOC_STATS=1) ----------------------------------------------------------------
//
// The global operator new / delete, as the default ones (malloc, or _aligned_malloc for over-aligned types),
// so all C++ in the program goes through them. With RT_ALLOC_STATS=1 every allocation of 256 KB or more is
// counted with its four innermost callers (offsets in nxrt.exe; resolve with a linker map), and a thread
// prints the last second's count, bytes and busiest call sites. The UCRT heap hands blocks this large to the
// OS on free, so each new one is fresh memory: one demand-zero fault per 4 KB when it is written.

#include <cstdlib>
#include <new>

namespace {

constexpr size_t kBigAlloc = 256 << 10;
constexpr int kSites = 256, kFrames = 4;

struct AllocSite {
    std::atomic<uint64_t> key{0}; // hash of the frames (0 = free)
    uintptr_t frames[kFrames]{};
    std::atomic<uint64_t> n{0}, bytes{0};
};
AllocSite g_sites[kSites];
std::atomic<uint64_t> g_big_n{0}, g_big_bytes{0};
std::atomic<int> g_alloc_stats{-1}; // -1 unknown, 0 off, 1 on
uintptr_t g_exe_base;

void AllocReport() {
    uint64_t last_n = 0, last_b = 0, last_site[kSites] = {};
    for (uint64_t sec = 1;; ++sec) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const uint64_t n = g_big_n.load(), b = g_big_bytes.load();
        if (n == last_n) continue;
        struct Row { uint64_t dn; int i; } rows[kSites];
        int m = 0;
        for (int i = 0; i < kSites; ++i) {
            const uint64_t sn = g_sites[i].n.load();
            if (sn != last_site[i]) rows[m++] = {sn - last_site[i], i};
            last_site[i] = sn;
        }
        std::sort(rows, rows + m, [](const Row& a, const Row& c) { return a.dn > c.dn; });
        rt::platform::LockStderr();
        std::fprintf(stderr, "[rt] allocs: t=%llu s: %llu of >= 256 KB, %.1f MB;", (unsigned long long)sec,
                     (unsigned long long)(n - last_n), (b - last_b) / 1048576.0);
        for (int r = 0; r < std::min(m, 4); ++r) {
            const AllocSite& s = g_sites[rows[r].i];
            std::fprintf(stderr, " %llux at", (unsigned long long)rows[r].dn);
            for (uintptr_t f : s.frames) {
                if (f) std::fprintf(stderr, " +0x%llx", (unsigned long long)(f - g_exe_base));
            }
            std::fprintf(stderr, ";");
        }
        std::fprintf(stderr, "\n");
        rt::platform::UnlockStderr();
        last_n = n, last_b = b;
    }
}

bool AllocStatsOn() {
    int s = g_alloc_stats.load(std::memory_order_relaxed);
    if (s >= 0) return s == 1;
    const char* e = std::getenv("RT_ALLOC_STATS");
    s = (e && *e && *e != '0') ? 1 : 0;
    int expected = -1;
    if (g_alloc_stats.compare_exchange_strong(expected, s) && s == 1) {
        g_exe_base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
        std::thread(AllocReport).detach();
    }
    return g_alloc_stats.load() == 1;
}

void CountBig(size_t size) {
    void* frames[kFrames + 2] = {};
    const USHORT got = RtlCaptureStackBackTrace(2, kFrames, frames, nullptr); // skip CountBig and operator new
    uint64_t key = 1469598103934665603ULL;
    for (USHORT i = 0; i < got; ++i) key = (key ^ reinterpret_cast<uintptr_t>(frames[i])) * 1099511628211ULL;
    key |= 1;
    g_big_n.fetch_add(1, std::memory_order_relaxed);
    g_big_bytes.fetch_add(size, std::memory_order_relaxed);
    for (int probe = 0; probe < kSites; ++probe) {
        AllocSite& s = g_sites[(key + probe) % kSites];
        uint64_t k = s.key.load(std::memory_order_acquire);
        if (k == 0) {
            uint64_t zero = 0;
            if (s.key.compare_exchange_strong(zero, key)) {
                for (USHORT i = 0; i < got; ++i) s.frames[i] = reinterpret_cast<uintptr_t>(frames[i]);
                k = key;
            } else {
                k = zero;
            }
        }
        if (k == key) {
            s.n.fetch_add(1, std::memory_order_relaxed);
            s.bytes.fetch_add(size, std::memory_order_relaxed);
            return;
        }
    }
}

void* Allocate(size_t size, size_t align) {
    if (size >= kBigAlloc && AllocStatsOn()) CountBig(size);
    for (;;) {
        void* p = align > __STDCPP_DEFAULT_NEW_ALIGNMENT__ ? _aligned_malloc(size ? size : 1, align)
                                                            : std::malloc(size ? size : 1);
        if (p) return p;
        std::new_handler h = std::get_new_handler();
        if (!h) return nullptr;
        h();
    }
}

} // namespace

void* operator new(size_t n) {
    if (void* p = Allocate(n, 0)) return p;
    throw std::bad_alloc();
}
void* operator new[](size_t n) {
    if (void* p = Allocate(n, 0)) return p;
    throw std::bad_alloc();
}
void* operator new(size_t n, const std::nothrow_t&) noexcept { return Allocate(n, 0); }
void* operator new[](size_t n, const std::nothrow_t&) noexcept { return Allocate(n, 0); }
void* operator new(size_t n, std::align_val_t a) {
    if (void* p = Allocate(n, static_cast<size_t>(a))) return p;
    throw std::bad_alloc();
}
void* operator new[](size_t n, std::align_val_t a) {
    if (void* p = Allocate(n, static_cast<size_t>(a))) return p;
    throw std::bad_alloc();
}
void* operator new(size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { return Allocate(n, static_cast<size_t>(a)); }
void* operator new[](size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { return Allocate(n, static_cast<size_t>(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t a) noexcept {
    if (static_cast<size_t>(a) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) _aligned_free(p); else std::free(p);
}
void operator delete[](void* p, std::align_val_t a) noexcept {
    if (static_cast<size_t>(a) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) _aligned_free(p); else std::free(p);
}
void operator delete(void* p, size_t, std::align_val_t a) noexcept {
    if (static_cast<size_t>(a) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) _aligned_free(p); else std::free(p);
}
void operator delete[](void* p, size_t, std::align_val_t a) noexcept {
    if (static_cast<size_t>(a) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) _aligned_free(p); else std::free(p);
}
void operator delete(void* p, std::align_val_t a, const std::nothrow_t&) noexcept {
    if (static_cast<size_t>(a) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) _aligned_free(p); else std::free(p);
}
void operator delete[](void* p, std::align_val_t a, const std::nothrow_t&) noexcept {
    if (static_cast<size_t>(a) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) _aligned_free(p); else std::free(p);
}
