#!/bin/bash
# The game builder's tools, as a release ships them (runtime/src/build/build.h): <out>/tools/,
# beside nxrt, where the builder looks for them.
#   nxrecomp[.exe]               the lifter (Windows: cross-built here with tools/cross, static C runtime)
#   llvm/bin/                    clang + the linker - Windows: clang.exe + lld-link.exe from tools/llvm-win (the
#                                official LLVM 23.1.2 Windows release, tools/llvm-win/SOURCE.txt); Linux: clang + ld.lld
#                                from tools/llvm
#   llvm/lib/clang/23/include/   only the headers the module's sources include (a probe: clang -M over all of a lift's
#                                sources), as LASRecomp's tools/zig_slim.sh keeps only what its probe includes
#   game/                        the module's own sources (runtime/src/game); Windows: kernel32.lib, made here from
#                                kernel32.def with llvm-dlltool (the one import, game/libc.c)
#   mods/                        the exefs mods that ship: the 60 fps patch (Fl4sh_#9174, public/NOTICE.md)
#   hactool[.exe]                unpacking the player's .nsp files with their keys (third_party/hactool, ISC, with its
#                                mbedtls, Apache-2.0; built by tools/hactool-build)
#   LICENSES/                    LLVM's, hactool's and mbedtls's licences
# Then the check: the probe's sources compile with the slim copy alone (Linux: the copy's clang; Windows: the Linux
# clang of the same release against the copy's headers - the same files).
#
# Usage: tools/make_toolchain.sh <out dir> [windows|linux] [gen dir for the probe, default build/m3/gen_101]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:?out dir}" && cd "$1" && pwd)
TARGET=${2:-windows}
GEN=$(cd "${3:-$REPO/build/m3/gen_101}" && pwd)
LLVM=$REPO/tools/llvm/bin
T=$OUT/tools
FRAME_RATE_MOD="[60FPS v1.0.1]" # (runtime/src/mods.h kFrameRateMod)
rm -rf "$T"
mkdir -p "$T/llvm/bin" "$T/game" "$T/mods" "$T/LICENSES"

# the lifter
case $TARGET in
windows)
    cmake -S "$REPO/lifter" -B "$REPO/build/win/lifter" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="$REPO/tools/cross/windows-clang-cl.cmake" -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded > /dev/null
    ninja -C "$REPO/build/win/lifter" -j8 nxrecomp > /dev/null
    cp "$REPO/build/win/lifter/nxrecomp.exe" "$T/"
    cmake -S "$REPO/tools/hactool-build" -B "$REPO/build/win/hactool" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="$REPO/tools/cross/windows-clang-cl.cmake" -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded > /dev/null
    ninja -C "$REPO/build/win/hactool" -j8 > /dev/null
    cp "$REPO/build/win/hactool/hactool.exe" "$T/"
    W=$REPO/tools/llvm-win
    cp "$W/bin/clang.exe" "$W/bin/lld-link.exe" "$T/llvm/bin/"
    RES=$W/lib/clang/23/include ;;
linux)
    ninja -C "$REPO/build/lifter" nxrecomp > /dev/null
    cp "$REPO/build/lifter/nxrecomp" "$T/"
    cmake -S "$REPO/tools/hactool-build" -B "$REPO/build/hactool" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$LLVM/clang" > /dev/null
    ninja -C "$REPO/build/hactool" -j8 > /dev/null
    cp "$REPO/build/hactool/hactool" "$T/"
    cp -L "$LLVM/clang" "$LLVM/ld.lld" "$T/llvm/bin/"
    RES=$("$LLVM/clang" -print-resource-dir)/include ;;
*) echo "target: windows or linux"; exit 2 ;;
esac

# the module's sources (and its import library)
cp -r "$REPO/runtime/src/game/." "$T/game/"
[ "$TARGET" = windows ] && "$LLVM/llvm-dlltool" -d "$T/game/kernel32.def" -l "$T/game/kernel32.lib" -m i386:x86-64
cp -r "$REPO/mods/$FRAME_RATE_MOD" "$T/mods/"
cp "$REPO/tools/llvm-win/LICENSE.TXT" "$T/LICENSES/LLVM.txt"
cp "$REPO/third_party/hactool/LICENSE" "$T/LICENSES/hactool.txt"
cp "$REPO/third_party/hactool/mbedtls/apache-2.0.txt" "$T/LICENSES/mbedtls.txt"

# clang's headers: the ones the probe includes (the same 23.1.2 headers on both hosts)
G=$REPO/runtime/src/game
probe=("$GEN"/funcs_*.c "$GEN/nx_table.c" "$GEN/recomp_runtime.c" "$G/entry.c" "$G/libc.c")
flags=(-std=c11 -w -msse4.2 -ffreestanding -nostdlibinc -isystem "$G/freestanding" -I "$GEN" -I "$G" -DNDEBUG)
[ "$TARGET" = windows ] && flags+=(--target=x86_64-pc-windows-msvc) || flags+=(--target=x86_64-unknown-linux-gnu)
LRES=$("$LLVM/clang" -print-resource-dir)/include
printf '%s\0' "${probe[@]}" | xargs -0 -n 64 "$LLVM/clang" "${flags[@]}" -M 2>/dev/null |  # (one at a time: parallel runs mix their lines)
    tr ' \\' '\n\n' | grep "^$LRES/" | sort -u | sed "s|^$LRES/||" > "$OUT/headers.txt"
mkdir -p "$T/llvm/lib/clang/23/include"
(cd "$RES" && xargs -a "$OUT/headers.txt" cp --parents -t "$T/llvm/lib/clang/23/include/")
echo "headers kept: $(wc -l < "$OUT/headers.txt") of $(find "$RES" -type f | wc -l)"

# the check: every probe source compiles against the slim headers alone
CHECK_CLANG=$LLVM/clang
[ "$TARGET" = linux ] && CHECK_CLANG=$T/llvm/bin/clang
printf '%s\0' "$GEN/funcs_0000.c" "$GEN/nx_table.c" "$GEN/recomp_runtime.c" "$G/entry.c" "$G/libc.c" |
    xargs -0 -P 4 -I{} "$CHECK_CLANG" "${flags[@]}" -resource-dir "$T/llvm/lib/clang/23" -fsyntax-only {}
echo "check: the probe compiles with the slim headers"
du -sh "$T" "$T"/* | sed "s|$OUT/||"
