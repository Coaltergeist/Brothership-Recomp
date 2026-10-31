// The crash report of runtime/src/platform/platform.h (InstallCrashReport), on whichever implementation it is
// built with (posix.cpp, windows.cpp): one crash per run, chosen by the argument — "fault" (a write to an
// unmapped address), "throw" (an exception nobody catches: std::terminate), "abort". tests/platform/run.sh
// checks that each run ends with exit code 71 and a report: "[rt] CRASH:", a host stack, the describe
// callback's line (and on Windows nxrt_crash.dmp).
#include "platform.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {

void Describe() {
    std::fprintf(stderr, "[rt]   describe: called\n");
}

[[gnu::noinline]] void Fault() {
    volatile int* p = reinterpret_cast<volatile int*>(static_cast<uintptr_t>(0x10));
    *p = 1;
}

[[gnu::noinline]] void Throw() {
    throw std::runtime_error("nobody catches this");
}

} // namespace

int main(int argc, char** argv) {
    rt::platform::InstallCrashReport(Describe);
    const char* mode = argc > 1 ? argv[1] : "";
    if (std::strcmp(mode, "fault") == 0) Fault();
    else if (std::strcmp(mode, "throw") == 0) Throw();
    else if (std::strcmp(mode, "abort") == 0) std::abort();
    std::fprintf(stderr, "usage: crash_test fault|throw|abort\n");
    return 2;
}
