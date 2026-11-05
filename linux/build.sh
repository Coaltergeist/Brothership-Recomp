#!/bin/sh
# Builds the Linux release's programs (the AppImage) in a copy of the
# sources that tools/package_appimage.sh staged: the tracked files only, llvm/ (clang + ld.lld + clang's headers,
# LLVM 23.1.2; lld's ICU 70 libraries and their licence), mods/ (the 60 fps patch) and DRIPPU_COMMIT. Run in that
# copy; installs into <prefix> (the AppDir's usr/):
#   bin/brothership-recomp   the runtime without game code (NXRT_GAME=module), as the Windows zip's exe
#   bin/tools/               the game builder's tools, where the runtime looks for them (beside the program), laid
#                            out as tools/make_toolchain.sh does for Windows - with all of clang's headers: the slim
#                            copy's probe needs a lifted game, which never enters this build
#   share/                   the desktop file, icon, metainfo, licences
#
# Usage: sh linux/build.sh <prefix> [jobs]
set -eu
PREFIX=${1:?install prefix}
JOBS=${2:-8}
ID=io.github.Coaltergeist.Brothership_Recomp
B=${BUILD_DIR:-$PWD/_build} # (BUILD_DIR: outside the staged copy, kept between packagings)
CLANG=$PWD/llvm/bin/clang
# CMake 4 refuses projects declaring a minimum below 3.5 (drippu's sirit/SPIRV-Headers); the variable also reaches the
# nested configures (the glslang host tool), the submodules unmodified (CMake 3 ignores it)
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# the lifter, and the lift's shared headers (all the runtime needs of a lift: GuestContext's layout)
cmake -S lifter -B "$B/lifter" -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C "$B/lifter" -j"$JOBS" nxrecomp
"$B/lifter/nxrecomp" --headers-only -o "$B/gen"

# the runtime (C with the shipped clang, C++ with the host's GCC, as the Linux development build)
cmake -S runtime -B "$B/rt" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CLANG" \
    -DGEN_DIR="$B/gen" -DNXRT_GAME=module -DRT_DRIPPU_COMMIT="$(cat DRIPPU_COMMIT)"
ninja -C "$B/rt" -j"$JOBS" nxrt

# hactool
cmake -S tools/hactool-build -B "$B/hactool" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CLANG"
ninja -C "$B/hactool" -j"$JOBS"

# install
install -Dm755 "$B/rt/nxrt" "$PREFIX/bin/brothership-recomp"
T=$PREFIX/bin/tools
install -Dm755 "$B/lifter/nxrecomp" "$T/nxrecomp"
install -Dm755 "$B/hactool/hactool" "$T/hactool"
install -Dm755 llvm/bin/clang "$T/llvm/bin/clang"
install -Dm755 llvm/bin/ld.lld "$T/llvm/bin/ld.lld"
# (lld needs ICU 70, the LLVM release's build host's; other systems have another ICU: these beside it, its RUNPATH)
for f in llvm/lib/libicu*.so.70; do install -Dm644 "$f" "$T/llvm/lib/$(basename "$f")"; done
mkdir -p "$T/llvm/lib/clang/23" "$T/game" "$T/mods" "$T/LICENSES"
cp -r llvm/lib/clang/23/include "$T/llvm/lib/clang/23/"
cp -r runtime/src/game/. "$T/game/"
cp -r mods/. "$T/mods/"
install -Dm644 llvm/LICENSE.TXT "$T/LICENSES/LLVM.txt"
install -Dm644 llvm/ICU.txt "$T/LICENSES/ICU.txt"
install -Dm644 third_party/hactool/LICENSE "$T/LICENSES/hactool.txt"
install -Dm644 third_party/hactool/mbedtls/apache-2.0.txt "$T/LICENSES/mbedtls.txt"

install -Dm644 linux/$ID.desktop "$PREFIX/share/applications/$ID.desktop"
install -Dm644 linux/$ID.metainfo.xml "$PREFIX/share/metainfo/$ID.appdata.xml"
install -Dm644 linux/$ID.svg "$PREFIX/share/icons/hicolor/scalable/apps/$ID.svg"
install -Dm644 public/LICENSE "$PREFIX/share/licenses/$ID/LICENSE"
install -Dm644 public/NOTICE.md "$PREFIX/share/licenses/$ID/NOTICE.md"
