#!/bin/bash
# Builds writetrack_bench.exe (BUFFER_CACHE_PLAN Phase 0: write-tracking costs) with the project's Windows cross
# toolchain (tools/cross/windows-clang-cl.cmake's compiler, headers and libraries) into build/win/writetrack/, and,
# with "run", runs it on Windows through WSL interop.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
L=$REPO/tools/llvm/bin W=$REPO/tools/winsdk OUT=$REPO/build/win/writetrack
mkdir -p "$OUT"
inc=(); for d in msvc/include sdk/include/ucrt sdk/include/um sdk/include/shared; do inc+=(/imsvc "$W/$d"); done
"$L/clang-cl" --target=x86_64-pc-windows-msvc /O2 /EHsc /std:c++20 /MT "${inc[@]}" /c "$REPO/tools/writetrack/writetrack_bench.cpp" /Fo"$OUT/writetrack_bench.obj"
"$L/lld-link" /nologo /out:"$OUT/writetrack_bench.exe" "$OUT/writetrack_bench.obj" /libpath:"$W/msvc/lib/x64" /libpath:"$W/sdk/lib/ucrt/x64" /libpath:"$W/sdk/lib/um/x64" kernel32.lib
echo "built $OUT/writetrack_bench.exe"
if [ "$1" = run ]; then (cd /mnt/c && "$OUT/writetrack_bench.exe" | tr -d '\r'); fi
