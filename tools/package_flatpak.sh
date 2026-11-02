#!/bin/bash
# The Linux / Steam Deck download: dist/BrothershipRecomp-Linux.flatpak and the release's
# dist/BrothershipRecomp-Linux.zip (the bundle, README_LINUX.txt as README.txt, LICENSE, NOTICE.md), the counterpart of
# tools/package_windows.sh. Stages in build/flatpak/staged what flatpak/build.sh builds from, inside the Flatpak SDK:
#   the tracked files (working-tree versions) of the parts the build uses - never an ignored file, so no dump, key or
#   build output can get in;
#   llvm/        clang + ld.lld + all of clang's headers from tools/llvm (LLVM 23.1.2), LLVM's licence; lld's ICU 70
#                libraries from this host (Ubuntu 22.04, the release's build host's version) and their licence
#   mods/        the 60 fps patch (Fl4sh_#9174, public/NOTICE.md)
#   DRIPPU_COMMIT  drippu's commit (the copy has no .git; runtime/CMakeLists.txt RT_DRIPPU_COMMIT)
# then flatpak-builder (org.freedesktop.Sdk 26.08, installed per user) and the bundle, then the scan as the zip's.
#
# Usage: tools/package_flatpak.sh [jobs]   (default 8: the heat rule)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
JOBS=${1:-8}
ID=io.github.Coaltergeist.Brothership_Recomp
F=$REPO/build/flatpak
S=$F/staged
OUT=$REPO/dist/BrothershipRecomp-Linux.flatpak
FRAME_RATE_MOD="[60FPS v1.0.1]" # (runtime/src/mods.h kFrameRateMod)
OWN=(lifter runtime tools/hactool-build flatpak)
SUBS=(third_party/drippu third_party/fmt third_party/glslang third_party/boost third_party/SDL third_party/imgui
      third_party/ooz third_party/hactool)

# the sources
rm -rf "$S"
mkdir -p "$S"
copy() { # <dir> <file list on stdin, NUL-separated, relative to dir>
    mkdir -p "$S/$1"
    tar -C "$REPO/$1" --null -T - -cf - | tar -C "$S/$1" -xf -
}
git -C "$REPO" ls-files -z --cached --others --exclude-standard -- "${OWN[@]}" | copy .
P=$REPO/public; [ -d "$P" ] || P=$REPO # (the public repository keeps LICENSE and NOTICE.md at its root)
mkdir -p "$S/public" && cp "$P/LICENSE" "$P/NOTICE.md" "$S/public/"
for d in "${SUBS[@]}"; do git -C "$REPO/$d" ls-files -z --recurse-submodules | copy "$d"; done
L=$REPO/tools/llvm
mkdir -p "$S/llvm/bin" "$S/llvm/lib/clang/23" "$S/mods"
cp -L "$L/bin/clang" "$L/bin/ld.lld" "$S/llvm/bin/"
cp -r "$L/lib/clang/23/include" "$S/llvm/lib/clang/23/"
cp "$REPO/tools/llvm-win/LICENSE.TXT" "$S/llvm/LICENSE.TXT"
for lib in $(readelf -d "$L/bin/lld" | sed -n 's/.*Shared library: \[\(libicu[a-z0-9]*\.so\.[0-9]*\)\]/\1/p'); do
    cp -L "/lib/x86_64-linux-gnu/$lib" "$S/llvm/lib/$lib"
done
cp /usr/share/doc/libicu70/copyright "$S/llvm/ICU.txt"
cp -r "$REPO/mods/$FRAME_RATE_MOD" "$S/mods/"
git -C "$REPO/third_party/drippu" rev-parse --short=12 HEAD > "$S/DRIPPU_COMMIT"
cp "$REPO/flatpak/$ID.yml" "$F/"
echo "staged: $(du -sh "$S" | cut -f1)"

# the build (per-user SDK; no rofiles-fuse in WSL), then the bundle
flatpak-builder --user --force-clean --disable-rofiles-fuse --ccache --jobs="$JOBS" --state-dir="$F/state" --repo="$F/repo" \
    "$F/app" "$F/$ID.yml" > "$F/build.log" 2>&1 || { tail -30 "$F/build.log"; exit 1; }

# the scan: no game files, no key-like hex in text files (as tools/package_windows.sh; clang's own headers left out:
# opencl-c-base.h spells math constants in 30+ digits)
A=$F/app/files
bad=$(find "$A" -type f \( -iname '*.nsp' -o -iname '*.xci' -o -iname '*.nca' -o -iname '*.nso' -o -iname '*.npdm' \
      -o -iname '*.keys' -o -iname '*.utoc' -o -iname '*.ucas' -o -iname '*.pak' -o -iname 'image.bin' \) )
[ -z "$bad" ] || { echo "game files in the package: $bad"; exit 1; }
if grep -rIlE '\b[0-9a-fA-F]{32,}\b' "$A" --include='*.txt' --include='*.md' --include='*.pchtxt' --include='*.c' --include='*.h' \
    | grep -v -e '\.pchtxt$' -e '/tools/llvm/lib/clang/23/include/'; then echo "key-like hex in the files above"; exit 1; fi

mkdir -p "$REPO/dist"
# (--runtime-repo: installing the file fetches the 26.08 runtime from Flathub when the machine lacks it)
flatpak build-bundle --runtime-repo=https://dl.flathub.org/repo/flathub.flatpakrepo "$F/repo" "$OUT" "$ID"
# the download: the bundle with the readme and licences, in one folder as the Windows zip
python3 -I - "$OUT" "$REPO/dist/BrothershipRecomp-Linux.zip" "$P/README_LINUX.txt" "$P/LICENSE" "$P/NOTICE.md" <<'EOF2'
import os, sys, zipfile
bundle, out, readme, *rest = sys.argv[1:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    z.write(bundle, "BrothershipRecomp-Linux/" + os.path.basename(bundle))
    z.write(readme, "BrothershipRecomp-Linux/README.txt")
    for f in rest: z.write(f, "BrothershipRecomp-Linux/" + os.path.basename(f))
EOF2
du -sh "$A" "$OUT" "$REPO/dist/BrothershipRecomp-Linux.zip" | sed "s|$REPO/||"
