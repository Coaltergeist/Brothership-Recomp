// The game module's compile and link flags, as tools/build_game.sh has them (the reference:
// a change here goes there too). Part of the module id (NXRT_TOOLCHAIN_ID hashes this file): a change rebuilds every
// player's module.
#pragma once

#include <string>
#include <vector>

namespace rt::build {

/// Every source's flags, before the include folders (the builder adds -isystem <game>/freestanding, -I <gen>,
/// -I <game>). The lifted code as the static build has it (-O1, recomp_runtime.c -O2, SSE4.2), freestanding: only
/// clang's headers and the stand-ins. Unwind tables: a guest thread ends by a C++ exception thrown through the lifted
/// code (without them Windows ended the process with 0xE06D7363 and Linux in std::terminate, 2026-10-31).
inline std::vector<std::string> CommonFlags() {
    std::vector<std::string> f{"-std=c11", "-w", "-msse4.2", "-ffreestanding", "-nostdlibinc", "-DNDEBUG",
                               "-fasynchronous-unwind-tables"};
#ifdef _WIN32
    f.push_back("--target=x86_64-pc-windows-msvc");
#else
    for (const char* x : {"--target=x86_64-unknown-linux-gnu", "-fPIC", "-fvisibility=hidden"}) f.push_back(x);
#endif
    return f;
}

/// The optimisation level of one source (by file name).
inline const char* OptFlag(const std::string& file_name) {
    return file_name == "recomp_runtime.c" ? "-O2" : "-O1";
}

/// The link, before the objects: no C runtime (game/libc.c's entry point, its TLS directory), no default libraries,
/// one KERNEL32 import from game/kernel32.lib (Windows gives a DLL that imports nothing no TLS slot, 2026-10-31).
inline std::vector<std::string> LinkFlags(const std::string& out) {
#ifdef _WIN32
    return {"/dll", "/entry:_DllMainCRTStartup", "/nodefaultlib", "/machine:x64", "/opt:ref", "/swaprun:net", "/noimplib",
            "/out:" + out};
#else
    return {"--target=x86_64-unknown-linux-gnu", "-shared", "-nostdlib", "-fuse-ld=lld", "-o", out};
#endif
}

/// The lifter's options besides the inputs, -o and the mods (as tools/build_game.sh lifts).
inline std::vector<std::string> LiftFlags() {
    return {"--trace-pc=func"};
}

} // namespace rt::build
