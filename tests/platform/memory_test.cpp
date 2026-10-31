// The guest-memory contract of runtime/src/platform/platform.h, on whichever
// implementation it is built with (posix.cpp, windows.cpp): the window lies at
// the address asked for; a range is accessible only while committed (an access
// anywhere else faults) and reads as zero until written and again after
// ZeroWindow and a new commit, whatever path the implementation takes (Windows:
// small ranges zeroed in place and kept, large ones and the kept ones over the
// cap given back), device views alias the backing, and a guest thread touching
// memory during remaps sees its own data.
// tests/platform/run.sh builds and runs it on Linux and on Windows.
#include "platform.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <setjmp.h>
#include <signal.h>
#endif

using rt::platform::CommitWindow;
using rt::platform::ZeroWindow;

namespace {

constexpr size_t K = 0x1000, M = 1 << 20, G = 1ULL << 30;
int g_fail = 0;

#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);        \
            ++g_fail;                                               \
        }                                                           \
    } while (0)

#ifdef _WIN32
bool Faults(const uint8_t* p) { return IsBadReadPtr(p, 1); } // (SEH; windows.cpp's handler passes unmapped ones on)
#else
sigjmp_buf g_probe;
void OnProbeFault(int) { siglongjmp(g_probe, 1); }
/// Whether reading *p faults.
bool Faults(const uint8_t* p) {
    struct sigaction sa{}, old_segv{}, old_bus{};
    sa.sa_handler = OnProbeFault;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);
    volatile bool fault = true; // (volatile: it must survive the jump back from the handler)
    if (sigsetjmp(g_probe, 1) == 0) {
        (void)*static_cast<const volatile uint8_t*>(p);
        fault = false;
    }
    sigaction(SIGSEGV, &old_segv, nullptr);
    sigaction(SIGBUS, &old_bus, nullptr);
    return fault;
}
#endif

bool Is(const uint8_t* p, size_t n, uint8_t v) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] != v) return false;
    }
    return true;
}

} // namespace

int main() {
    rt::platform::InitProcess();
    constexpr uintptr_t kAt = 0x4000'0000'0000; // the runtime's (rt.h kHostWindowBase)
    uint8_t* w = rt::platform::ReserveWindow(16 * G, kAt);
    CHECK(w == reinterpret_cast<uint8_t*>(kAt));
    if (!w) return 1;
    CHECK(rt::platform::ReserveWindow(16 * G, kAt + 4 * G) == nullptr); // taken: refused, not moved elsewhere

    // Uncommitted memory faults; committed memory reads as zero and is writable.
    CHECK(Faults(w + 123) && Faults(w + 16 * G - 1));
    CommitWindow(w, 4 * K);
    CHECK(w[123] == 0);
    w[123] = 1;
    CHECK(w[123] == 1);
    CHECK(Faults(w + 4 * K));

    // A small ZeroWindow inside a committed range: zero there, neighbours kept; mapping it again stays zero.
    uint8_t* a = w + 64 * M;
    CommitWindow(a, 256 * K);
    std::memset(a, 0xAB, 256 * K);
    ZeroWindow(a + 4 * K, 8 * K);
    CHECK(a[4 * K - 1] == 0xAB);
    CHECK(Faults(a + 4 * K) && Faults(a + 12 * K - 1));
    CHECK(a[12 * K] == 0xAB);
    CommitWindow(a + 4 * K, 8 * K);
    CHECK(Is(a + 4 * K, 8 * K, 0));
    std::memset(a + 4 * K, 0xCD, 8 * K);
    CHECK(Is(a + 4 * K, 8 * K, 0xCD));
    CHECK(Is(a, 4 * K, 0xAB) && Is(a + 12 * K, 244 * K, 0xAB));

    // A large ZeroWindow (over the 4 MB in-place limit): zero, neighbours kept.
    uint8_t* b = w + 256 * M;
    CommitWindow(b, 16 * M);
    std::memset(b, 0x5A, 16 * M);
    ZeroWindow(b + 4 * K, 8 * M);
    CHECK(Is(b, 4 * K, 0x5A));
    CHECK(Faults(b + 4 * K) && Faults(b + 4 * K + 8 * M - 1));
    CHECK(Is(b + 4 * K + 8 * M, 16 * M - 8 * M - 4 * K, 0x5A));
    CommitWindow(b + 4 * K, 8 * M);
    CHECK(Is(b + 4 * K, 8 * M, 0));

    // A small ZeroWindow over committed and never-committed memory, then mapped again.
    uint8_t* c = w + 512 * M;
    CommitWindow(c, 64 * K);
    std::memset(c, 1, 64 * K);
    ZeroWindow(c, M);
    CHECK(Faults(c) && Faults(c + M - 1));
    CommitWindow(c, M);
    CHECK(Is(c, M, 0));
    c[M - 1] = 7;
    CHECK(c[M - 1] == 7);

    // Device views over a kept range; ZeroWindow over a device view; the backing keeps its bytes.
    uint8_t* dev = rt::platform::CreateDeviceBacking(64 * M);
    CHECK(dev != nullptr);
    uint8_t* d = w + 768 * M;
    CommitWindow(d, M);
    std::memset(d, 0x11, M);
    ZeroWindow(d, M);
    CommitWindow(d, M);
    CHECK(Is(d, M, 0));
    std::memset(d, 0x22, 64 * K);
    CHECK(rt::platform::MapDeviceView(d, 64 * K, 0));
    std::memset(dev, 0x33, 64 * K);
    CHECK(d[0] == 0x33 && d[64 * K - 1] == 0x33);
    d[5] = 0x44;
    CHECK(dev[5] == 0x44);
    CHECK(Is(d + 64 * K, M - 64 * K, 0));
    rt::platform::UnmapDeviceView(d, 64 * K);
    CHECK(Is(d, 64 * K, 0));
    ZeroWindow(d, M);
    CHECK(Faults(d) && Faults(d + M - 1));
    CHECK(rt::platform::MapDeviceView(d, 64 * K, 0));
    CHECK(d[5] == 0x44);
    rt::platform::UnmapDeviceView(d, 64 * K); // the runtime's order (rt.cpp UnmapRange): views first, then zero
    ZeroWindow(d, 128 * K);
    CHECK(Faults(d) && Faults(d + 128 * K - 1));
    CommitWindow(d, 128 * K);
    CHECK(Is(d, 128 * K, 0));
    CHECK(dev[5] == 0x44);

    // More than 1 GB of small ranges given back (Windows: the kept ones are released past the cap).
    uint8_t* e = w + 2 * G;
    for (int i = 0; i < 300; ++i) {
        uint8_t* p = e + i * 8 * M;
        CommitWindow(p, 4 * M);
        std::memset(p, i | 1, 4 * M);
        ZeroWindow(p, 4 * M);
    }
    for (int i = 0; i < 300; ++i) {
        uint8_t* p = e + i * 8 * M;
        CHECK(Faults(p) && Faults(p + 4 * M - 1));
        CommitWindow(p, 4 * M);
        CHECK(p[0] == 0 && p[4 * M - 1] == 0 && Is(p + 2 * M - 64 * K, 128 * K, 0));
        p[K] = 9;
        CHECK(p[K] == 9 && p[0] == 0);
    }

    // A guest thread writing its memory while another maps and gives back ranges nearby.
    std::atomic<bool> stop{false};
    std::atomic<int> bad{0}, bad_zero{0};
    CommitWindow(w + 6 * G, 256 * M);
    std::thread toucher([&] {
        uint8_t* t = w + 6 * G;
        for (size_t i = 0; !stop.load(std::memory_order_relaxed); i = (i + 1) % 4096) {
            uint8_t* p = t + i * 16 * K; // 64 KB blocks: 256 MB
            const uint8_t v = static_cast<uint8_t>(i | 1);
            p[0] = v;
            p[16 * K - 1] = v;
            if (p[0] != v || p[16 * K - 1] != v) bad.fetch_add(1);
        }
    });
    uint8_t* f = w + 6 * G + 256 * M; // past the toucher's 256 MB
    for (int i = 0; i < 20000; ++i) {
        uint8_t* p = f + (i % 64) * 16 * K;
        CommitWindow(p, 16 * K);
        p[0] = 1;
        ZeroWindow(p, 16 * K);
        CommitWindow(p, 16 * K);
        if (p[0] != 0) bad_zero.fetch_add(1);
    }
    stop = true;
    toucher.join();
    CHECK(bad.load() == 0);
    CHECK(bad_zero.load() == 0);

    std::printf(g_fail ? "memory_test: %d FAILED\n" : "memory_test: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
