// BUFFER_CACHE_PLAN Phase 0: what CPU write tracking costs on this PC, for the choice of §3.2 (Windows A: page
// protection + a vectored exception handler; Windows B: MEM_WRITE_WATCH + GetWriteWatch). A standalone Windows
// program, built in WSL with the project's cross toolchain (tools/writetrack/build.sh) and run on Windows.
//
// Memory as the runtime's guest memory: views of a pagefile-backed section (platform/windows.cpp maps them with
// MapViewOfFile3 onto placeholders; protection works the same on a plain view). Write watch needs private memory
// (VirtualAlloc with MEM_WRITE_WATCH), which is what option B would change the memory design to.
//
// Prints, per measurement, the median of several rounds in microseconds per page (or per call).
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr size_t kPage = 4096;
constexpr size_t kPages = 4096; // 16 MB: about the heavy scene's pages per frame (13-21 MB)
constexpr int kRounds = 7;

double Now() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Option A's handler: a write to a protected page of the tracked range makes it writable again and records it.
uint8_t* g_base = nullptr;
size_t g_size = 0;
std::atomic<size_t> g_faults{0};

LONG CALLBACK OnFault(EXCEPTION_POINTERS* e) {
    const EXCEPTION_RECORD* r = e->ExceptionRecord;
    if (r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2 || r->ExceptionInformation[0] != 1) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const auto at = reinterpret_cast<uint8_t*>(r->ExceptionInformation[1]);
    if (at < g_base || at >= g_base + g_size) return EXCEPTION_CONTINUE_SEARCH;
    DWORD old;
    VirtualProtect(reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(at) & ~(kPage - 1)), kPage, PAGE_READWRITE, &old);
    g_faults.fetch_add(1, std::memory_order_relaxed);
    return EXCEPTION_CONTINUE_EXECUTION;
}

void Touch(uint8_t* p, size_t first, size_t count) {
    for (size_t i = first; i < first + count; ++i) p[i * kPage] = static_cast<uint8_t>(p[i * kPage] + 1);
}

} // namespace

int main() {
    std::printf("writetrack_bench: %zu pages (%zu MB) a round, median of %d rounds\n", kPages, kPages * kPage >> 20, kRounds);

    // the runtime's kind of memory: a view of a pagefile-backed section
    const HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT, 0,
                                              static_cast<DWORD>(kPages * kPage), nullptr);
    auto* view = static_cast<uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kPages * kPage));
    if (!section || !view) {
        std::printf("section view failed: %lu\n", GetLastError());
        return 1;
    }
    std::memset(view, 1, kPages * kPage);

    std::vector<double> plain, copy, protect_each, protect_whole, fault1, fault4;
    std::vector<uint8_t> dst(kPage);
    g_base = view, g_size = kPages * kPage;
    AddVectoredExceptionHandler(1, OnFault);
    for (int round = 0; round < kRounds; ++round) {
        double t = Now();
        Touch(view, 0, kPages);
        plain.push_back((Now() - t) / kPages);

        t = Now();
        for (size_t i = 0; i < kPages; ++i) std::memcpy(dst.data(), view + i * kPage, kPage);
        copy.push_back((Now() - t) / kPages);

        DWORD old;
        t = Now();
        for (size_t i = 0; i < kPages; ++i) VirtualProtect(view + i * kPage, kPage, PAGE_READONLY, &old);
        protect_each.push_back((Now() - t) / kPages);
        VirtualProtect(view, kPages * kPage, PAGE_READWRITE, &old);

        t = Now();
        VirtualProtect(view, kPages * kPage, PAGE_READONLY, &old);
        protect_whole.push_back(Now() - t);

        g_faults = 0;
        t = Now();
        Touch(view, 0, kPages); // each page: fault, handler, VirtualProtect, retry
        fault1.push_back((Now() - t) / kPages);
        if (g_faults != kPages) std::printf("round %d: %zu faults, expected %zu\n", round, g_faults.load(), kPages);

        VirtualProtect(view, kPages * kPage, PAGE_READONLY, &old);
        g_faults = 0;
        t = Now();
        std::vector<std::thread> threads;
        for (size_t k = 0; k < 4; ++k) threads.emplace_back(Touch, view, k * kPages / 4, kPages / 4);
        for (auto& th : threads) th.join();
        fault4.push_back((Now() - t) / kPages); // wall time per page, four threads faulting at once
        if (g_faults != kPages) std::printf("round %d (4 threads): %zu faults, expected %zu\n", round, g_faults.load(), kPages);
        VirtualProtect(view, kPages * kPage, PAGE_READWRITE, &old);
    }
    std::printf("A  write a page, unprotected             %8.3f us/page\n", Median(plain));
    std::printf("   copy a 4 KB page (what a fault saves)  %8.3f us/page\n", Median(copy));
    std::printf("   VirtualProtect read-only, page by page %8.3f us/page\n", Median(protect_each));
    std::printf("   VirtualProtect read-only, 16 MB at once %7.1f us/call\n", Median(protect_whole));
    std::printf("   first write to a protected page (fault + handler + VirtualProtect + retry), 1 thread %8.3f us/page\n",
                Median(fault1));
    std::printf("   the same, 4 threads at once (wall time per page)                              %8.3f us/page\n",
                Median(fault4));

    // option B: private memory with write watch
    const size_t big = size_t{1} << 30; // GetWriteWatch over a region the size of a guest heap (1 GB reserved)
    auto* ww = static_cast<uint8_t*>(VirtualAlloc(nullptr, big, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE));
    if (!ww || !VirtualAlloc(ww, kPages * kPage, MEM_COMMIT, PAGE_READWRITE)) {
        std::printf("MEM_WRITE_WATCH failed: %lu\n", GetLastError());
        return 1;
    }
    std::memset(ww, 1, kPages * kPage);
    std::vector<void*> addrs(kPages);
    std::vector<double> ww_first, ww_query_dirty, ww_query_clean, ww_query_big;
    for (int round = 0; round < kRounds; ++round) {
        ULONG_PTR n = kPages;
        DWORD gran;
        GetWriteWatch(WRITE_WATCH_FLAG_RESET, ww, kPages * kPage, addrs.data(), &n, &gran); // clean start
        double t = Now();
        Touch(ww, 0, kPages);
        ww_first.push_back((Now() - t) / kPages);
        n = kPages;
        t = Now();
        GetWriteWatch(WRITE_WATCH_FLAG_RESET, ww, kPages * kPage, addrs.data(), &n, &gran);
        ww_query_dirty.push_back(Now() - t);
        if (n != kPages) std::printf("round %d: GetWriteWatch reported %llu pages, expected %zu\n", round,
                                     static_cast<unsigned long long>(n), kPages);
        n = kPages;
        t = Now();
        GetWriteWatch(WRITE_WATCH_FLAG_RESET, ww, kPages * kPage, addrs.data(), &n, &gran);
        ww_query_clean.push_back(Now() - t);
        Touch(ww, 0, kPages / 16);
        n = kPages;
        t = Now();
        GetWriteWatch(WRITE_WATCH_FLAG_RESET, ww, big, addrs.data(), &n, &gran);
        ww_query_big.push_back(Now() - t);
    }
    std::printf("B  first write to a watched page                       %8.3f us/page\n", Median(ww_first));
    std::printf("   GetWriteWatch + reset, 16 MB, all pages written     %8.1f us/call\n", Median(ww_query_dirty));
    std::printf("   GetWriteWatch + reset, 16 MB, none written          %8.1f us/call\n", Median(ww_query_clean));
    std::printf("   GetWriteWatch + reset, 1 GB reserved (16 MB committed, 1 MB written) %8.1f us/call\n",
                Median(ww_query_big));
    return 0;
}
