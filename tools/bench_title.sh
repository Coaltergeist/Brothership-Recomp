#!/bin/bash
# Brothership benchmark run (see handoff.md / README "Running Brothership"):
# title-screen route, a freeze watchdog (the runtime's own, RT_WATCHDOG: no
# frame for 30 s logs "watchdog:" and two thread dumps; the script then stops
# the run), and a benchmark summary (from fps.csv).
# Usage: tools/bench_title.sh <cap seconds> [VAR=value ...]   e.g. RT_TRACE_ROMFS=1
# No screenshots unless asked: CAPTURE=1 in the script's environment captures every 100 frames,
# or give RT_CAPTURE=<frame>[,<frame>...] (the given variables win over the script's, RT_INPUT too: e.g.
# RT_INPUT=60s~@Opening.bk2:A:10/100 lets the opening movie play instead of skipping it).
# Headless unless RT_HEADLESS=0 is given (a window and the FPS overlay).
# Another build: NXRT=<binary> GEN=<nxrecomp output dir> in the script's environment (relative to build/m3;
# default ./rt_101/nxrt and gen_101, game version 1.0.1), e.g. NXRT=./rt/nxrt GEN=gen NXRT_GAME=$PWD/extracted
# tools/bench_title.sh 200 for the 1.0.0 build
# Other game data: NXRT_GAME=<folder with exefs/main.npdm and romfs/> (absolute; default extracted/v101).
cap=${1:-720}; shift
cd "$(dirname "$0")/../build/m3" || exit 1
pgrep -x nxrt >/dev/null && { echo "nxrt already running"; exit 1; }
rm -f frame_*.png run_vk.log fps.csv
caps=(); [ "${CAPTURE:-0}" != 0 ] && caps=("RT_CAPTURE=1,$(seq -s, 100 100 80000)")
env RT_HEADLESS=1 RT_INPUT=60s~@Opening.bk2:A:10/100,@Opening.bk2:B:1000000 "${caps[@]}" "$@" RT_GPU_RENDERER=vulkan \
    RT_ROMFS=${NXRT_GAME:-$PWD/../../extracted/v101}/romfs \
    RT_NPDM=${NXRT_GAME:-$PWD/../../extracted/v101}/exefs/main.npdm "${NXRT:-./rt_101/nxrt}" "${GEN:-gen_101}/image.bin" "${GEN:-gen_101}/symbols.txt" > run_vk.log 2>&1 &
P=$!
start=$(date +%s); froze=0
while kill -0 $P 2>/dev/null; do
    sleep 5
    now=$(date +%s)
    if grep -q "watchdog: no frame" run_vk.log; then
        grep -m1 "watchdog: no frame" run_vk.log
        sleep 5; kill $P; froze=1; break # both dumps are written within 4 s
    fi
    if [ $((now - start)) -ge "$cap" ]; then kill $P; break; fi
done
wait $P 2>/dev/null; code=$?
echo "exit=$code frames=$(grep -c 'binder transaction 7' run_vk.log) wall=$(( $(date +%s) - start )) s froze=$froze"
grep -m1 "threads:" run_vk.log
grep "video_core up\|hid: event .*bk2\|break:\|FATAL" run_vk.log | head -20 | cut -c1-200
python3 -I - <<'EOF'
import csv, os, re
T=None
for l in open('run_vk.log',errors='replace'):
    m=re.search(r'hid: event .*Title_Mv.* \(t=([\d.]+) s\)',l)
    if m: T=float(m.group(1)); break
rows=[(int(r['second']),int(r['frames']),r) for r in csv.DictReader(open('fps.csv'))] if os.path.exists('fps.csv') else []
total=0; at={}
for sec,n,_ in rows: # the second in which the 1000th / 5000th frame was presented
    total+=n
    for k in (1000,5000):
        if total>=k and k not in at: at[k]=sec
print('loading 1000..5000:', f'frames 1000..5000 in {at[5000]-at[1000]} s = {4000/max(1,at[5000]-at[1000]):.1f} fps'
      if len(at)==2 else 'n/a')
if T is None: print('no title event')
else:
    w=[r for sec,_,r in rows if T+5<=sec<T+45]
    if w:
        n=sum(int(r['frames']) for r in w); p95=sorted(float(r['p95_ms'] or 0) for r in w)[len(w)//2]
        print(f'title at t={T:.1f} s; benchmark (fps.csv, 5-45 s after): {n/len(w):.1f} fps over {len(w)} s, '
              f'median p95 frame time {p95:.1f} ms, worst frame {max(float(r["max_ms"] or 0) for r in w):.1f} ms')
    else: print(f'title at t={T:.1f} s; no fps.csv rows 5-45 s after')
EOF
