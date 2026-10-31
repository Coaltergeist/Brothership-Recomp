#!/usr/bin/env bash
# Builds tools/devkitpro-compat: a DEVKITPRO root whose devkitA64 runs on
# glibc < 2.36 (Ubuntu 22.04). devkitPro's GCC 16 binaries need glibc 2.36 only
# for arc4random. The patched copy gets that one function from a small shim
# library; everything else is symlinked to the untouched /opt/devkitpro.
#
# Usage: tools/devkitpro_compat/make_compat.sh   (then export DEVKITPRO=$PWD/tools/devkitpro-compat)
# Needs: /opt/devkitpro with switch-dev installed, patchelf in tools/venv.
set -euo pipefail

SRC=${DEVKITPRO_SRC:-/opt/devkitpro}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DEST=$REPO/tools/devkitpro-compat
PATCHELF=$REPO/tools/venv/bin/patchelf

[ -x "$PATCHELF" ] || { echo "missing $PATCHELF (tools/venv/bin/pip install patchelf)"; exit 1; }

rm -rf "$DEST"
mkdir -p "$DEST"
for entry in "$SRC"/*; do
    name=$(basename "$entry")
    case $name in
        devkitA64|pacman) ;;
        *) ln -s "$entry" "$DEST/$name" ;;
    esac
done
cp -a "$SRC/devkitA64" "$DEST/devkitA64"

mkdir -p "$DEST/devkitA64/compat"
gcc -O2 -shared -fPIC "$HERE/arc4random.c" -o "$DEST/devkitA64/compat/libarc4random.so"

# True if the file's version requirements mention $2. Counts instead of using
# grep -q: under pipefail, grep -q exiting early fails the pipeline on SIGPIPE.
needs_version() {
    [ "$(readelf -V "$1" 2>/dev/null | grep -c "$2")" -gt 0 ]
}

patched=()
while IFS= read -r -d '' f; do
    if needs_version "$f" 'GLIBC_2\.36'; then
        rel=$(realpath --relative-to="$(dirname "$f")" "$DEST/devkitA64/compat")
        "$PATCHELF" --clear-symbol-version arc4random --add-needed libarc4random.so \
            --add-rpath "\$ORIGIN/$rel" "$f"
        patched+=("$f")
    fi
done < <(find "$DEST/devkitA64" -type f -perm -u+x -print0)

python3 -I "$HERE/retarget_verneed.py" GLIBC_2.36 GLIBC_2.34 "${patched[@]}" >/dev/null
echo "patched ${#patched[@]} binaries"

while IFS= read -r -d '' f; do
    if needs_version "$f" 'GLIBC_2\.3[6-9]'; then
        echo "error: $f still requires glibc >= 2.36"
        exit 1
    fi
done < <(find "$DEST/devkitA64" -type f -perm -u+x -print0)
"$DEST/devkitA64/bin/aarch64-none-elf-gcc" --version | head -1
echo "export DEVKITPRO=$DEST"
