#!/usr/bin/env bash
# M1 end-to-end test: build the libnx hello-world example, lift it, build the
# runtime, run it with a scripted Plus press, and check the result.
#
# Needs devkitPro switch-dev. Set DEVKITPRO (on Ubuntu 22.04 use the overlay
# from tools/devkitpro_compat/make_compat.sh).
# M1_LIFT_FLAGS: extra nxrecomp flags (e.g. --no-fp-helpers); M1_OUT: another output directory
# The lift is the checked one (--no-direct-mem): the program draws through the driver seam, which a
# direct lift does not support (rt.cpp).
# (default build/tests/m1, which tests/m1/run_win.sh reads).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${M1_OUT:-$REPO/build/tests/m1}
: "${DEVKITPRO:?set DEVKITPRO (see README: devkitPro)}"
mkdir -p "$OUT"
step() { echo "== $*"; }

step "build templates/application"
rm -rf "$OUT/application"
cp -r "$DEVKITPRO/examples/switch/templates/application" "$OUT/"
make -s -C "$OUT/application" > "$OUT/make.log" 2>&1

step "build lifter"
cmake -S "$REPO/lifter" -B "$OUT/lifter" -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$OUT/lifter" > "$OUT/lifter.log" 2>&1

step "lift (validating discovery against ELF symbols)"
rm -rf "$OUT/gen"
"$OUT/lifter/nxrecomp" "$OUT/application/application.elf" -o "$OUT/gen" --validate --no-direct-mem ${M1_LIFT_FLAGS:-} | tee "$OUT/lift.log"
grep -Eq 'recall \(symbols found as function starts\): ([0-9]+)/\1$' "$OUT/lift.log" \
    || { echo "FAIL: discovery missed function symbols"; exit 1; }

step "build runtime"
cmake -S "$REPO/runtime" -B "$OUT/rt" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGEN_DIR="$OUT/gen" > /dev/null
cmake --build "$OUT/rt" > "$OUT/rt.log" 2>&1

step "run (Plus pressed at frame 3)"
cd "$OUT"
rm -f frame_*.png
status=0
RT_HEADLESS=1 RT_PRESS_PLUS_AT=3 RT_CAPTURE=1 timeout 60 "$OUT/rt/nxrt" gen/image.bin gen/symbols.txt > run.log 2>&1 || status=$?
[ "$status" -eq 0 ] || { tail -5 run.log; echo "FAIL: runtime exited with $status"; exit 1; }
grep -q 'am: application requested exit' run.log || { echo "FAIL: no clean applet exit"; exit 1; }
echo "exit OK: $(grep -c 'svc ' run.log) SVCs, $(grep -c 'ipc  ' run.log) IPC log lines"

step "check frame"
python3 -I "$REPO/tests/m1/check_frame.py" frame_1.png
echo "M1 PASS"
