#!/bin/bash
# The Linux / Steam Deck download: dist/BrothershipRecomp-x86_64.AppImage and the release's
# dist/BrothershipRecomp-Linux.zip (the AppImage, README_LINUX.txt as README.txt, LICENSE, NOTICE.md), the counterpart
# of tools/package_windows.sh. Built on this host (Ubuntu 22.04, glibc 2.35: the AppImage runs on systems at least as
# new). Stages in build/appimage/staged what linux/build.sh builds from:
#   the tracked files (working-tree versions) of the parts the build uses - never an ignored file, so no dump, key or
#   build output can get in;
#   llvm/        clang + ld.lld + all of clang's headers from tools/llvm (LLVM 23.1.2), LLVM's licence; lld's ICU 70
#                libraries from this host (Ubuntu 22.04, the release's build host's version) and their licence
#   mods/        the 60 fps patch (Fl4sh_#9174, public/NOTICE.md)
#   DRIPPU_COMMIT  drippu's commit (the copy has no .git; runtime/CMakeLists.txt RT_DRIPPU_COMMIT)
# then the build (objects kept in build/appimage/obj), the AppDir, the scan as the zip's, the library check (nothing
# newer than glibc 2.35 / this host's libstdc++), and the AppImage: the AppImage project's type-2 static runtime
# (tools/appimage/runtime-x86_64, the user's download: needs only fusermount3 on the player's system) followed by the
# AppDir as a squashfs (zstd).
#
# Usage: tools/package_appimage.sh [jobs]   (default 8: the heat rule; APPDIR_ONLY=1: stop before the AppImage)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
JOBS=${1:-8}
ID=io.github.Coaltergeist.Brothership_Recomp
F=$REPO/build/appimage
S=$F/staged
D=$F/AppDir
RUNTIME=$REPO/tools/appimage/runtime-x86_64
OUT=$REPO/dist/BrothershipRecomp-x86_64.AppImage
FRAME_RATE_MOD="[60FPS v1.0.1]" # (runtime/src/mods.h kFrameRateMod)
OWN=(lifter runtime tools/hactool-build linux)
SUBS=(third_party/drippu third_party/fmt third_party/glslang third_party/boost third_party/SDL third_party/imgui
      third_party/ooz third_party/hactool)
[ -f "$RUNTIME" ] || [ "${APPDIR_ONLY:-0}" = 1 ] || { echo "no $RUNTIME: the AppImage project's type-2 static runtime (github.com/AppImage/type2-runtime, releases: runtime-x86_64)"; exit 1; }

# the sources (tar keeps the files' times: an unchanged file is not rebuilt)
rm -rf "$S"
mkdir -p "$S"
copy() { # <dir> <file list on stdin, NUL-separated, relative to dir>
    mkdir -p "$S/$1"
    tar -C "$REPO/$1" --null -T - -cf - | tar -C "$S/$1" -xf -
}
git -C "$REPO" ls-files -z --cached --others --exclude-standard -- "${OWN[@]}" | copy .
P=$REPO/public; [ -d "$P" ] || P=$REPO # (the public repository keeps LICENSE and NOTICE.md at its root)
mkdir -p "$S/public" && cp -p "$P/LICENSE" "$P/NOTICE.md" "$S/public/"
for d in "${SUBS[@]}"; do git -C "$REPO/$d" ls-files -z --recurse-submodules | copy "$d"; done
L=$REPO/tools/llvm
mkdir -p "$S/llvm/bin" "$S/llvm/lib/clang/23" "$S/mods"
cp -pL "$L/bin/clang" "$L/bin/ld.lld" "$S/llvm/bin/"
cp -pr "$L/lib/clang/23/include" "$S/llvm/lib/clang/23/"
cp -p "$REPO/tools/llvm-win/LICENSE.TXT" "$S/llvm/LICENSE.TXT"
for lib in $(readelf -d "$L/bin/lld" | sed -n 's/.*Shared library: \[\(libicu[a-z0-9]*\.so\.[0-9]*\)\]/\1/p'); do
    cp -pL "/lib/x86_64-linux-gnu/$lib" "$S/llvm/lib/$lib"
done
cp -p /usr/share/doc/libicu70/copyright "$S/llvm/ICU.txt"
cp -pr "$REPO/mods/$FRAME_RATE_MOD" "$S/mods/"
git -C "$REPO/third_party/drippu" rev-parse --short=12 HEAD > "$S/DRIPPU_COMMIT"
echo "staged: $(du -sh "$S" | cut -f1)"

# the build and the AppDir
rm -rf "$D"
(cd "$S" && BUILD_DIR="$F/obj" sh linux/build.sh "$D/usr" "$JOBS") > "$F/build.log" 2>&1 || { tail -30 "$F/build.log"; exit 1; }
install -m755 "$S/linux/AppRun" "$D/AppRun"
cp "$S/linux/$ID.desktop" "$D/$ID.desktop"
cp "$S/linux/$ID.svg" "$D/$ID.svg"
ln -s "$ID.svg" "$D/.DirIcon"
# SDL's configure summary: the backends it can load on the player's system (sound, display, controllers)
sdl=$(grep -E "^-- +SDL_(PIPEWIRE|PULSEAUDIO|ALSA|WAYLAND|WAYLAND_LIBDECOR|X11|DBUS|LIBUDEV|HIDAPI) +\(Wanted" "$F/build.log" \
    | sed -E 's/^-- +(SDL_[A-Z_]+) .*: (ON|OFF) *$/\1=\2/' | tr '\n' ' ')
echo "SDL: $sdl"
case " $sdl " in *=OFF*) echo "  a sound, display or controller backend is off (its dev package is missing on this host)"; exit 1 ;; esac

# the scan: no game files, no key-like hex in text files (as tools/package_windows.sh; clang's own headers left out:
# opencl-c-base.h spells math constants in 30+ digits)
bad=$(find "$D" -type f \( -iname '*.nsp' -o -iname '*.xci' -o -iname '*.nca' -o -iname '*.nso' -o -iname '*.npdm' \
      -o -iname '*.keys' -o -iname '*.utoc' -o -iname '*.ucas' -o -iname '*.pak' -o -iname 'image.bin' \) )
[ -z "$bad" ] || { echo "game files in the package: $bad"; exit 1; }
if grep -rIlE '\b[0-9a-fA-F]{32,}\b' "$D" --include='*.txt' --include='*.md' --include='*.pchtxt' --include='*.c' --include='*.h' \
    | grep -v -e '\.pchtxt$' -e '/llvm/lib/clang/23/include/'; then echo "key-like hex in the files above"; exit 1; fi

# the libraries: each program's newest glibc / libstdc++ symbol version, and nothing outside the system's base set
for f in "$D/usr/bin/brothership-recomp" "$D/usr/bin/tools/nxrecomp" "$D/usr/bin/tools/hactool" \
         "$D/usr/bin/tools/llvm/bin/clang" "$D/usr/bin/tools/llvm/bin/ld.lld"; do
    g=$(objdump -T "$f" 2>/dev/null | { grep -oE 'GLIBC_[0-9.]+' || true; } | sort -uV | tail -1)
    x=$(objdump -T "$f" 2>/dev/null | { grep -oE 'GLIBCXX_[0-9.]+' || true; } | sort -uV | tail -1)
    n=$(readelf -d "$f" 2>/dev/null | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | tr '\n' ' ')
    echo "${f#"$D/"}: ${g:-no glibc} ${x:-} needs: $n"
    case $g in GLIBC_2.3[6-9]* | GLIBC_2.[4-9]*) echo "  newer than glibc 2.35"; exit 1 ;; esac
    for lib in $n; do
        case $lib in libc.so.6 | libm.so.6 | libdl.so.2 | libpthread.so.0 | librt.so.1 | libstdc++.so.6 | libgcc_s.so.1 | \
            ld-linux-x86-64.so.2 | libz.so.1 | liblzma.so.5 | libicu*.so.70) ;;
            *) echo "  unexpected library $lib"; exit 1 ;;
        esac
    done
done

[ "${APPDIR_ONLY:-0}" = 1 ] && { du -sh "$D" | sed "s|$REPO/||"; exit 0; }

# the AppImage: the runtime, then the AppDir as a squashfs (as appimagetool makes it)
mkdir -p "$REPO/dist"
rm -f "$F/AppDir.squashfs"
mksquashfs "$D" "$F/AppDir.squashfs" -root-owned -noappend -comp zstd -Xcompression-level 19 -b 1M -quiet > /dev/null
cat "$RUNTIME" "$F/AppDir.squashfs" > "$OUT"
chmod 755 "$OUT"
# the download: the AppImage with the readme and licences, in one folder as the Windows zip
python3 -I - "$OUT" "$REPO/dist/BrothershipRecomp-Linux.zip" "$P/README_LINUX.txt" "$P/LICENSE" "$P/NOTICE.md" <<'EOF2'
import os, sys, zipfile
app, out, readme, *rest = sys.argv[1:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    i = zipfile.ZipInfo.from_file(app, "BrothershipRecomp-Linux/" + os.path.basename(app))
    i.external_attr = 0o100755 << 16  # executable when unzipped
    i.compress_type = zipfile.ZIP_DEFLATED
    with open(app, "rb") as f: z.writestr(i, f.read())
    z.write(readme, "BrothershipRecomp-Linux/README.txt")
    for f in rest: z.write(f, "BrothershipRecomp-Linux/" + os.path.basename(f))
EOF2
du -sh "$D" "$OUT" "$REPO/dist/BrothershipRecomp-Linux.zip" | sed "s|$REPO/||"
