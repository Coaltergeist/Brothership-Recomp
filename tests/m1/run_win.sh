#!/usr/bin/env bash
# M1 on Windows (PORT_PLAN Phase 4): the runtime cross-built in WSL (tools/cross) with the code
# tests/m1/run.sh lifted (run that first), run on Windows from \\wsl.localhost through WSL interop,
# then the same checks as run.sh.
#
# Needs the cross toolchain in tools/llvm and tools/winsdk (README "Windows build").
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$REPO/build/tests/m1
[ -f "$OUT/gen/image.bin" ] || { echo "run tests/m1/run.sh first (it lifts the program)"; exit 1; }
step() { echo "== $*"; }

step "build runtime for Windows"
cmake -S "$REPO/runtime" -B "$REPO/build/win/m1" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE="$REPO/tools/cross/windows-clang-cl.cmake" -DGEN_DIR="$OUT/gen" > /dev/null
cmake --build "$REPO/build/win/m1" --target nxrt > "$OUT/rt_win.log" 2>&1

step "run on Windows (Plus pressed at frame 3)"
unc() { echo "\\\\wsl.localhost\\${WSL_DISTRO_NAME:-Ubuntu}$(echo "$1" | tr / '\\')"; }
cat > "$OUT/run_win.ps1" <<EOF
Set-Location '$(unc "$OUT")'
Remove-Item -ErrorAction SilentlyContinue frame_*.png
\$env:RT_HEADLESS = '1'; \$env:RT_PRESS_PLUS_AT = '3'; \$env:RT_CAPTURE = '1'
\$p = Start-Process -FilePath '$(unc "$REPO/build/win/m1/nxrt.exe")' -ArgumentList 'gen/image.bin', 'gen/symbols.txt' \`
     -NoNewWindow -PassThru -RedirectStandardOutput run_win.out -RedirectStandardError run_win.log
if (-not \$p.WaitForExit(60000)) { \$p.Kill(); 'TIMEOUT after 60 s' }
EOF
(cd /mnt/c && powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(unc "$OUT/run_win.ps1")" | tr -d '\r')
cd "$OUT"
grep -q 'am: application requested exit' run_win.log || { tail -5 run_win.log; echo "FAIL: no clean applet exit"; exit 1; }
grep -q 'guest exited with code 0' run_win.log || { tail -5 run_win.log; echo "FAIL: no clean exit"; exit 1; }
echo "exit OK: $(grep -c 'svc ' run_win.log) SVCs, $(grep -c 'ipc  ' run_win.log) IPC log lines"

step "check frame"
python3 -I "$REPO/tests/m1/check_frame.py" frame_1.png
echo "M1 (Windows) PASS"
