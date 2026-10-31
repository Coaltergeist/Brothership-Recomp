#!/usr/bin/env bash
# The platform layer's tests (guest memory, crash report) on Linux (posix.cpp) and on Windows (windows.cpp, cross-built in WSL
# and run from \\wsl.localhost through interop, as tests/m1/run_win.sh does). Seconds to build, seconds to run.
# The Windows run needs the cross toolchain (README "Windows build"); skip it with NO_WIN=1.
# It touches ~1.2 GB of memory for a moment (the kept-range cap check).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC=$REPO/tests/platform
step() { echo "== $*"; }
# crash_check <host> <mode> <exit code> <report file> [minidump]: one crash_test run's result.
crash_check() {
    local host=$1 mode=$2 code=$3 report=$4 dump=${5:-}
    local ok=1
    [ "$code" = 71 ] || ok=0
    grep -q "^\[rt\] CRASH: " "$report" || ok=0
    grep -q "describe: called" "$report" || ok=0
    [ "$(grep -cE "crash_test" "$report")" -ge 2 ] || ok=0 # stack frames in the test program
    [ -z "$dump" ] || [ -s "$dump" ] || ok=0
    if [ $ok = 1 ]; then echo "crash_test $mode: PASS ($(grep -m1 "CRASH: " "$report" | cut -c13-90))"
    else echo "FAIL: crash_test $mode on $host (exit code $code)"; cat "$report"; exit 1; fi
}

step "Linux (posix.cpp)"
cmake -S "$SRC" -B "$REPO/build/tests/platform" -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$REPO/build/tests/platform" -j2 > /dev/null
"$REPO/build/tests/platform/memory_test"
for mode in fault throw abort; do
    code=0; "$REPO/build/tests/platform/crash_test" $mode 2> "$REPO/build/tests/platform/crash_$mode.txt" || code=$?
    crash_check Linux $mode $code "$REPO/build/tests/platform/crash_$mode.txt"
done

[ "${NO_WIN:-0}" = 1 ] && exit 0

step "Windows (windows.cpp)"
OUT=$REPO/build/win/platform_test
cmake -S "$SRC" -B "$OUT" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE="$REPO/tools/cross/windows-clang-cl.cmake" > /dev/null
cmake --build "$OUT" -j2 > /dev/null
unc() { echo "\\\\wsl.localhost\\${WSL_DISTRO_NAME:-Ubuntu}$(echo "$1" | tr / '\\')"; }
cat > "$OUT/run.ps1" <<EOF
\$p = Start-Process -FilePath '$(unc "$OUT/memory_test.exe")' -NoNewWindow -PassThru -Wait \`
     -RedirectStandardOutput '$(unc "$OUT/run.out")'
exit \$p.ExitCode
EOF
status=0
(cd /mnt/c && powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(unc "$OUT/run.ps1")" | tr -d '\r') || status=$?
tr -d '\r' < "$OUT/run.out"
[ "$status" = 0 ] || { echo "FAIL: Windows exit code $status"; exit 1; }

for mode in fault throw abort; do
    rm -f "$OUT/nxrt_crash.dmp"
    cat > "$OUT/crash.ps1" <<PS1
\$p = Start-Process -FilePath '$(unc "$OUT/crash_test.exe")' -ArgumentList '$mode' -WorkingDirectory '$(unc "$OUT")' \`
     -NoNewWindow -PassThru -Wait -RedirectStandardError '$(unc "$OUT/crash_$mode.txt")'
exit \$p.ExitCode
PS1
    code=0
    (cd /mnt/c && powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(unc "$OUT/crash.ps1")" | tr -d '\r') || code=$?
    tr -d '\r' < "$OUT/crash_$mode.txt" > "$OUT/crash_$mode.lf" && mv "$OUT/crash_$mode.lf" "$OUT/crash_$mode.txt"
    crash_check Windows $mode $code "$OUT/crash_$mode.txt" "$OUT/nxrt_crash.dmp"
done
