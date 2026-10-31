#!/bin/bash
# Brothership's title route on Windows (PORT_PLAN Phase 4/5): tools/bench_title.sh's run, with the
# cross-built build/win/bro_101/nxrt.exe (game version 1.0.1) started on Windows from \\wsl.localhost through WSL interop.
# Game data and the generated image stay in WSL (the build's GEN_DIR, extracted/v101/); the log, fps.csv and
# frame_<n>.png go to build/win/run/.
# Usage: tools/bench_title_win.sh <cap seconds> [VAR=value ...]   e.g. RT_GPU=nvn RT_HEADLESS=0
# Headless unless RT_HEADLESS=0 is given (a window and the FPS overlay on the Windows desktop).
# No screenshots unless asked, as bench_title.sh: CAPTURE=1 in the environment, or RT_CAPTURE=... given.
cap=${1:-720}; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
EXE=${NXRT_EXE:-$REPO/build/win/bro_101/nxrt.exe} # (NXRT_EXE: another build, e.g. build/win/fps60_101)
# NXRT_RUN: the run folder (its save/ folder is the game's save; default build/win/run). A named one keeps each
# session's log, fps.csv and captures in sessions/<date_time>/ afterwards (long play sessions, e.g. a new game).
RUN=${NXRT_RUN:-$REPO/build/win/run}
case $RUN in /*) ;; *) RUN=$REPO/$RUN ;; esac
[ -f "$EXE" ] || { echo "no $EXE: build it first (handoff.md, Windows build)"; exit 1; }
# The image and symbols of the lift the build was compiled from (its GEN_DIR; NXRT_GEN overrides), and
# the game data it belongs to (NXRT_GAME: a folder with exefs/main.npdm and romfs/; 1.0.0's is extracted/).
GEN=${NXRT_GEN:-$(sed -n 's/^GEN_DIR:PATH=//p' "$(dirname "$EXE")/CMakeCache.txt" 2>/dev/null)}
GEN=${GEN:-$REPO/build/m3/gen_101}
GAME=${NXRT_GAME:-$REPO/extracted/v101}
echo "image: $GEN, game data: $GAME"
mkdir -p "$RUN" && cd "$RUN" || exit 1
rm -f frame_*.png run_vk.log run_vk.out fps.csv targets.csv
unc() { echo "\\\\wsl.localhost\\${WSL_DISTRO_NAME:-Ubuntu}$(echo "$1" | tr / '\\')"; }

# The variables, as bench_title.sh sets them (later ones win, in the same order as there).
caps=(); [ "${CAPTURE:-0}" != 0 ] && caps=("RT_CAPTURE=1,$(seq -s, 100 100 80000)")
# The PC's pipeline cache (the seed's ~1.1 GB, README) lives on the local disk, %LOCALAPPDATA%\brothership-recomp\cache
# (set in run_win.ps1 below): through \\wsl.localhost the file reads at ~29 MB/s. A given RT_PIPELINE_CACHE wins.
vars=(RT_HEADLESS=1 "RT_INPUT=60s~@Opening.bk2:A:10/100,@Opening.bk2:B:1000000" "${caps[@]}" "$@"
      RT_GPU_RENDERER=vulkan "RT_ROMFS=$(unc "$GAME/romfs")"
      "RT_NPDM=$(unc "$GAME/exefs/main.npdm")")
{
    echo "Set-Location '$(unc "$RUN")'"
    echo "\$env:RT_PIPELINE_CACHE = Join-Path \$env:LOCALAPPDATA 'brothership-recomp\\cache'"
    for v in "${vars[@]}"; do echo "\$env:${v%%=*} = '${v#*=}'"; done
    cat <<EOF
\$p = Start-Process -FilePath '$(unc "$EXE")' -ArgumentList '$(unc "$GEN/image.bin")', '$(unc "$GEN/symbols.txt")' \`
     -NoNewWindow -PassThru -RedirectStandardOutput run_vk.out -RedirectStandardError run_vk.log
\$start = Get-Date; \$froze = 0
while (-not \$p.HasExited) {
    Start-Sleep -Seconds 5
    if (Select-String -Path run_vk.log -Pattern 'watchdog: no frame' -Quiet) {
        (Select-String -Path run_vk.log -Pattern 'watchdog: no frame' | Select-Object -First 1).Line
        Start-Sleep -Seconds 5; Stop-Process -Id \$p.Id -Force; \$froze = 1; break # both dumps within 4 s
    }
    if (((Get-Date) - \$start).TotalSeconds -ge $cap) { Stop-Process -Id \$p.Id -Force; break }
}
"wall=\$([int]((Get-Date) - \$start).TotalSeconds) s froze=\$froze"
EOF
} > run_win.ps1
(cd /mnt/c && powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(unc "$RUN/run_win.ps1")" | tr -d '\r')
tr -d '\r' < run_vk.log > run_vk.lf && mv run_vk.lf run_vk.log
echo "frames=$(grep -c 'binder transaction 7' run_vk.log)"
grep -m1 "threads:" run_vk.log
grep -m1 "Vulkan device" run_vk.log
grep "video_core up\|hid: event .*bk2\|break:\|FATAL" run_vk.log | head -20 | cut -c1-200
grep "^\[rt\] FLAG " run_vk.log | cut -c1-120 # the player's flags (F7 / Capture button)
# the same summary as tools/bench_title.sh
sed -n '/^python3 -I - <<.EOF.$/,/^EOF$/p' "$REPO/tools/bench_title.sh" | sed '1d;$d' | python3 -I -
if [ -n "$NXRT_RUN" ]; then
    s=sessions/$(date +%Y-%m-%d_%H%M) && mkdir -p "$s"
    mv run_vk.log run_vk.out fps.csv waits.csv heavy_*.csv targets.csv frame_*.png "$s"/ 2>/dev/null
    [ -d f9 ] && mv f9 "$s"/
    echo "session files: $RUN/$s"
fi
