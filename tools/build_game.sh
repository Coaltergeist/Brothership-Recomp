#!/bin/bash
# The game module from a lift, with nothing but clang and lld: the commands the start screen's
# builder will run on a player's machine (Phase 3). No Windows SDK, MinGW or glibc headers: the lifted code is built
# freestanding (clang's own headers + runtime/src/game/freestanding/), the module needs no C library
# (runtime/src/game/libc.c). Load it with RT_GAME_MODULE=<file> or put it beside an NXRT_GAME=module nxrt.
#
# Usage: tools/build_game.sh <gen dir> <out dir> [windows|linux] [jobs]
#   windows (default): <out>/nxgame.dll; linux: <out>/libnxgame.so. jobs: default half the cores (the heat rule).
#   Objects stay in <out>/obj: a second run compiles only sources newer than their object (or after a flags change).
#   NXGAME_SOURCES=<file ...>: compile only these (a probe; no link).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
GEN=$(cd "${1:?gen dir}" && pwd)
OUT=${2:?out dir}
TARGET=${3:-windows}
JOBS=${4:-$(( ($(nproc) + 1) / 2 ))}
LLVM=${LLVM:-$REPO/tools/llvm/bin}
GAME=$REPO/runtime/src/game
mkdir -p "$OUT/obj"
OUT=$(cd "$OUT" && pwd)

# The generated code's flags as the static build has them (runtime/CMakeLists.txt: -O1, recomp_runtime.c -O2,
# SSE4.2), plus freestanding: only clang's headers and the stand-ins.
# Unwind tables on both: a guest thread ends by a C++ exception thrown through the lifted code (clang-cl's default,
# not clang's when freestanding: without them Windows ended the process with 0xE06D7363 and Linux in std::terminate,
# 2026-10-31).
common=(-std=c11 -w -msse4.2 -ffreestanding -nostdlibinc -isystem "$GAME/freestanding" -I "$GEN" -I "$GAME" -DNDEBUG
        -fasynchronous-unwind-tables)
case $TARGET in
windows) common+=(--target=x86_64-pc-windows-msvc) ;;
linux) common+=(--target=x86_64-unknown-linux-gnu -fPIC -fvisibility=hidden) ;;
*) echo "target: windows or linux"; exit 2 ;;
esac
# the flags in a response file (paths with spaces); a change of flags compiles everything again
rsp=$(for a in "${common[@]}"; do printf '"%s"\n' "${a//\\/\\\\}"; done)
if [ "$(cat "$OUT/obj/flags.rsp" 2>/dev/null)" != "$rsp" ]; then rm -f "$OUT"/obj/*.o; printf '%s\n' "$rsp" > "$OUT/obj/flags.rsp"; fi

sources=("$GEN"/funcs_*.c "$GEN/nx_table.c" "$GEN/recomp_runtime.c" "$GAME/entry.c" "$GAME/libc.c")
[ -n "${NXGAME_SOURCES:-}" ] && read -r -a sources <<< "$NXGAME_SOURCES"
compile() { # <source>: <out>/obj/<name>.o unless up to date
    local src=$1 obj="$OUT/obj/$(basename "${1%.c}").o" opt=-O1
    [ "$(basename "$src")" = recomp_runtime.c ] && opt=-O2
    [ "$obj" -nt "$src" ] && return 0
    "$LLVM/clang" "@$OUT/obj/flags.rsp" "$opt" -c "$src" -o "$obj.tmp" && mv "$obj.tmp" "$obj"
}
export -f compile
export LLVM OUT
start=$(date +%s)
printf '%s\0' "${sources[@]}" | xargs -0 -P "$JOBS" -I{} bash -c 'compile "$1"' _ {}
echo "compiled ${#sources[@]} sources in $(( $(date +%s) - start )) s with $JOBS jobs"
[ -n "${NXGAME_SOURCES:-}" ] && exit 0

objs=("$OUT"/obj/*.o)
case $TARGET in
windows)
    # no C runtime: our own entry point (libc.c: the loader sets up TLS only for a DLL with one), no default
    # libraries; /SWAPRUN:NET as nxrt's (started from \\wsl.localhost)
    # (one KERNEL32 import, from our own .def: the loader sets up TLS only for a DLL that imports something, libc.c)
    "$LLVM/llvm-dlltool" -d "$GAME/kernel32.def" -l "$OUT/obj/kernel32.lib" -m i386:x86-64
    "$LLVM/lld-link" /dll /entry:_DllMainCRTStartup /nodefaultlib /machine:x64 /opt:ref /swaprun:net /out:"$OUT/nxgame.dll" \
        /implib:"$OUT/obj/nxgame.lib" "${objs[@]}" "$OUT/obj/kernel32.lib"
    echo "imports (only KERNEL32 GetCurrentThreadId expected):"; "$LLVM/llvm-readobj" --coff-imports "$OUT/nxgame.dll" | { grep -E "Name:|Symbol:" || true; }
    ls -la "$OUT/nxgame.dll" ;;
linux)
    "$LLVM/clang" --target=x86_64-unknown-linux-gnu -shared -nostdlib -fuse-ld=lld -o "$OUT/libnxgame.so" "${objs[@]}"
    echo "imports (only the dynamic loader's __tls_get_addr expected):"; "$LLVM/llvm-nm" -D --undefined-only "$OUT/libnxgame.so"
    ls -la "$OUT/libnxgame.so" ;;
esac
