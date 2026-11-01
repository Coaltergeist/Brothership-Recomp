#!/bin/bash
# The Windows download: dist/BrothershipRecomp-Windows.zip with
#   BrothershipRecomp.exe  the runtime (nxrt) without game code, no console window, Microsoft's C/C++ runtime linked
#                       in (/MT): built here in build/win/release (NXRT_GAME=module, NXRT_CONSOLE=OFF); it starts itself
#                       again by its own path (the seed stage, the restart into the game), so the name is free
#   background.png      the start screen's picture, if public/background.png exists (art of our own only, never the
#                       game's)
#   tools/              the game builder's tools (tools/make_toolchain.sh: lifter, clang + lld-link + headers, the
#                       module's sources, the 60 fps patch, LLVM's licence)
#   LICENSE, NOTICE.md  public/ (GPL-3.0-or-later; every library and the 60 fps patch credited)
#   README.txt          for testers (public/README_WINDOWS.txt)
# Nothing of the game and no key: the archive is scanned for game file types and key-like hex before it is written.
#
# Usage: tools/package_windows.sh [jobs]   (jobs for the runtime's build, default 8: the heat rule)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
JOBS=${1:-8}
B=$REPO/build/win/release
STAGE=$REPO/dist/BrothershipRecomp-Windows
ZIP=$REPO/dist/BrothershipRecomp-Windows.zip

# the runtime (configured once; the lift's headers only: GuestContext's layout)
if [ ! -f "$B/build.ninja" ]; then
    cmake -S "$REPO/runtime" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="$REPO/tools/cross/windows-clang-cl.cmake" -DGEN_DIR="$REPO/build/m3/gen_101" \
        -DNXGEN_OPT=/clang:-O1 -DNXRT_GAME=module -DNXRT_CONSOLE=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded > /dev/null
fi
ninja -C "$B" -j"$JOBS" nxrt > "$B/package_build.log" 2>&1 || { grep -E "error|FAILED" "$B/package_build.log" | tail -20; exit 1; }
imports=$("$REPO/tools/llvm/bin/llvm-readobj" --coff-imports "$B/nxrt.exe" | sed -n 's/^ *Name: //p' | sort -u)
if grep -qiE "^(msvcp|vcruntime|api-ms-win-crt)" <<< "$imports"; then
    echo "nxrt.exe still imports Microsoft's C runtime DLLs:"; grep -iE "^(msvcp|vcruntime|api-ms-win-crt)" <<< "$imports"; exit 1
fi

# the folder, then the archive
rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"
cp "$B/nxrt.exe" "$STAGE/BrothershipRecomp.exe"
[ -f "$REPO/public/background.png" ] && cp "$REPO/public/background.png" "$STAGE/"
"$REPO/tools/make_toolchain.sh" "$STAGE" windows > /dev/null
rm -f "$STAGE/headers.txt"
cp "$REPO/public/LICENSE" "$REPO/public/NOTICE.md" "$STAGE/"
cp "$REPO/public/README_WINDOWS.txt" "$STAGE/README.txt"
# the scan: no game files, no key-like hex in text files
bad=$(find "$STAGE" -type f \( -iname '*.nsp' -o -iname '*.xci' -o -iname '*.nca' -o -iname '*.nso' -o -iname '*.npdm' \
      -o -iname '*.keys' -o -iname '*.utoc' -o -iname '*.ucas' -o -iname '*.pak' -o -iname 'image.bin' \) )
[ -z "$bad" ] || { echo "game files in the package: $bad"; exit 1; }
if grep -rIlE '\b[0-9a-fA-F]{32,}\b' "$STAGE" --include='*.txt' --include='*.md' --include='*.pchtxt' --include='*.c' --include='*.h' \
    | grep -v '\.pchtxt$'; then echo "key-like hex in the files above"; exit 1; fi
python3 -I - "$STAGE" "$ZIP" <<'EOF'
import os, sys, zipfile
stage, out = sys.argv[1], sys.argv[2]
root = os.path.basename(stage)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for d, _, files in os.walk(stage):
        for f in sorted(files):
            p = os.path.join(d, f)
            z.write(p, os.path.join(root, os.path.relpath(p, stage)))
EOF
echo "BrothershipRecomp.exe imports: $(tr '\n' ' ' <<< "$imports")"
du -sh "$STAGE" "$ZIP" | sed "s|$REPO/||"
