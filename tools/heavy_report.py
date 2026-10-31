#!/usr/bin/env python3
"""HEAVY_FRAMES_PLAN Phase 0: the report of a run with RT_PROFILE_HEAVY (runtime/src/nvn/heavy.h).

Usage: tools/heavy_report.py <session or run folder> [--top N] [--cut 2,4,8] [--submit-ms 0.1]

Reads heavy_subs.csv, heavy_stretches.csv, heavy_dispatches.csv and waits.csv from the folder and prints, for each
heavy submission (translation or GPU time at RT_PROFILE_HEAVY's threshold or more):
  - its frame (the longest gap between presents while it was in flight), translation, GPU, driver time;
  - where the translation went: per record kind, per phase and part (draw / dispatch), the Vulkan functions;
  - the GPU: compute shaders against everything else (copies, barriers, draws);
  - its dispatches: CPU / driver / GPU time each, bytes filled and copied, write-backs and how much of them the
    shader changed (a write-back with 0 changed bytes is one the shader did not need: Phase 2's first candidate);
  - Phase 1's cut simulated: parts ended at a stretch with no open render pass once a part's translation passes
    each limit; each part's GPU work starts when it is submitted or when the previous part's ends.
Then the same per-dispatch phase averages for the light submissions of the same seconds, for comparison.
"""
import argparse
import collections
import csv
import os
import sys


def read(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path} (a run with RT_PROFILE_HEAVY and RT_TRACE_WAITS writes it)")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--cut", default="2,4,8")
    ap.add_argument("--submit-ms", type=float, default=0.1, help="cost of one more vkQueueSubmit, assumed")
    a = ap.parse_args()
    d = a.folder
    subs = read(os.path.join(d, "heavy_subs.csv"))
    stretches = read(os.path.join(d, "heavy_stretches.csv"))
    dispatches = read(os.path.join(d, "heavy_dispatches.csv"))
    waits = read(os.path.join(d, "waits.csv")) if os.path.exists(os.path.join(d, "waits.csv")) else []

    rows = collections.defaultdict(list)
    for r in subs:
        rows[r["sub"]].append(r)
    heavy = [s for s in rows if not s.startswith("s")]
    seconds = {s: v for s, v in rows.items() if s.startswith("s")}
    st_by = collections.defaultdict(list)
    for r in stretches:
        st_by[r["sub"]].append(r)
    di_by = collections.defaultdict(list)
    for r in dispatches:
        di_by[r["sub"]].append(r)

    # frames: presents, and each submission's [call, retire] from waits.csv
    presents = sorted(fnum(r["t_ms"]) for r in waits if r["kind"] == "present")
    span = {}
    for r in waits:
        if r["kind"] == "submit":
            span.setdefault(r["what"], [None, None])[0] = fnum(r["t_ms"])
        elif r["kind"] == "retire":
            span.setdefault(r["what"], [None, None])[1] = fnum(r["t_ms"])

    def frame_of(sub):
        lo, hi = span.get(sub, [None, None])
        if lo is None or hi is None or len(presents) < 2:
            return None
        best = 0.0
        for p, q in zip(presents, presents[1:]):
            if q >= lo and p <= hi:
                best = max(best, q - p)
        return best

    def val(rs, kind, name):
        for r in rs:
            if r["kind"] == kind and r["name"] == name:
                return fnum(r["ms"]), int(fnum(r["count"]))
        return 0.0, 0

    cuts = [float(x) for x in a.cut.split(",") if x]
    print(f"{len(heavy)} heavy submissions, {len(seconds)} seconds of light ones\n")
    disp_phase_heavy = collections.Counter()
    disp_n_heavy = 0
    for sub in sorted(heavy, key=lambda s: fnum(rows[s][0]["t_ms"])):
        rs = rows[sub]
        t = fnum(rs[0]["t_ms"]) / 1e3
        tr, recs = val(rs, "sub", "translate")
        gpu, _ = val(rs, "sub", "gpu")
        drv, calls = val(rs, "sub", "driver_total")
        over, _ = val(rs, "sub", "timer_overhead_est")
        shaders, ndisp = val(rs, "gpu", "dispatch_shaders")
        fr = frame_of(sub)
        print(f"=== {sub} at t={t:.2f} s: {recs} records; translation {tr:.1f} ms, GPU {gpu:.1f} ms"
              + (f", longest frame while in flight {fr:.1f} ms" if fr is not None else ""))
        print(f"  driver: {drv:.1f} ms in {calls} Vulkan calls ({drv / tr * 100 if tr else 0:.0f} % of translation;"
              f" timers' own cost ~{over:.1f} ms of that)")
        print(f"  GPU: compute shaders {shaders:.1f} ms in {ndisp} dispatches, the rest {gpu - shaders:.1f} ms"
              " (copies, barriers, draws)")
        for kind in ("dispatch", "draw", "other"):
            ms, n = val(rs, "records", kind)
            if not n:
                continue
            ph = sorted(((fnum(r["ms"]), r["name"]) for r in rs if r["kind"] == "phase." + kind), reverse=True)
            in_ph = sum(m for m, _ in ph)
            print(f"  {kind}: {n} records, {ms:.1f} ms ({ms / n * 1e3:.1f} us each); phases: "
                  + ", ".join(f"{name} {m:.1f}" for m, name in ph[:6]) + f"; outside phases {ms - in_ph:.1f}")
            parts = sorted(((fnum(r["ms"]), r["name"]) for r in rs if r["kind"] == "part." + kind), reverse=True)
            if parts:
                print("    parts: " + ", ".join(f"{name} {m:.1f}" for m, name in parts[:6]))
            if kind == "dispatch":
                disp_n_heavy += n
                for m, name in ph:
                    disp_phase_heavy[name] += m
                disp_phase_heavy["(all)"] += ms
        fns = sorted(((fnum(r["ms"]), int(fnum(r["count"])), r["name"]) for r in rs if r["kind"] == "driver"), reverse=True)
        print("  Vulkan functions: " + ", ".join(f"{name} {m:.1f} ms ({c})" for m, c, name in fns[:a.top]))

        ds = di_by.get(sub, [])
        if ds:
            def tot(k):
                return sum(fnum(r[k]) for r in ds)
            wb, ch = tot("wb_bytes"), tot("wb_changed")
            with_wb = [r for r in ds if fnum(r["wb_bytes"]) > 0]
            unchanged = [r for r in with_wb if fnum(r["wb_changed"]) == 0]
            gpu_us = [fnum(r["gpu_us"]) for r in ds if fnum(r["gpu_us"], -1) >= 0]
            mb = 1048576
            print(f"  dispatches: {len(ds)}; per dispatch CPU {tot('cpu_us') / len(ds):.1f} us (driver"
                  f" {tot('driver_us') / len(ds):.1f} us, {tot('driver_calls') / len(ds):.1f} calls), GPU"
                  f" {sum(gpu_us) / len(gpu_us) if gpu_us else 0:.1f} us")
            print(f"    filled for writes {tot('write_fills') / mb:.1f} MB ({tot('writes'):.0f} writes), reads filled"
                  f" {tot('read_fills') / mb:.1f} MB, from earlier GPU results {tot('gpu_parts') / mb:.1f} MB;"
                  f" to device {tot('to_device') / mb:.1f} MB, back {tot('from_device') / mb:.1f} MB")
            print(f"    written back {wb / mb:.1f} MB, of it changed {ch / mb:.2f} MB ({ch / wb * 100 if wb else 0:.1f} %);"
                  f" {len(unchanged)} of {len(with_wb)} dispatches with write-backs changed nothing"
                  f" (their write-backs {sum(fnum(r['wb_bytes']) for r in unchanged) / mb:.1f} MB)")
            by_prog = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
            for r in ds:
                e = by_prog[r["hash"]]
                e[0] += 1
                e[1] += fnum(r["cpu_us"])
                e[2] += max(fnum(r["gpu_us"], 0), 0)
                e[3] += fnum(r["wb_bytes"])
                e[4] += fnum(r["wb_changed"])
            top = sorted(by_prog.items(), key=lambda kv: -kv[1][1])[:a.top]
            print("    by program (CPU ms, GPU ms, written back MB, changed %): " + "; ".join(
                f"{h[-8:]} x{e[0]} {e[1] / 1e3:.1f}/{e[2] / 1e3:.1f}/{e[3] / mb:.1f}/{e[4] / e[3] * 100 if e[3] else 0:.0f}%"
                for h, e in top))

        st = st_by.get(sub, [])
        if st:
            cpu = [fnum(r["cpu_end_ms"]) for r in st]
            g = [max(fnum(r["gpu_ms"]), 0) for r in st]
            safe = [r["pass_open_end"] == "0" for r in st]
            base = cpu[-1] + a.submit_ms + sum(g)
            out = [f"none {base:.1f} ms"]
            for limit in cuts:
                start_cpu, gpu_end, parts, acc = 0.0, 0.0, 0, 0.0
                for i in range(len(st)):
                    acc += g[i]
                    last = i == len(st) - 1
                    if last or (cpu[i] - start_cpu >= limit and safe[i]):
                        submit = cpu[i] + a.submit_ms * (parts + 1)
                        gpu_end = max(submit, gpu_end) + acc
                        acc, start_cpu, parts = 0.0, cpu[i], parts + 1
                out.append(f"{limit:g} ms: {gpu_end:.1f} ms ({parts} parts)")
            closed = sum(safe) / len(safe) * 100
            print(f"  cut simulation (translation start to GPU end; {closed:.0f} % of stretches end with no open pass): "
                  + ", ".join(out))
        print()

    # the light submissions' dispatches, per second, for comparison
    light = collections.Counter()
    light_n = 0
    for s, rs in seconds.items():
        ms, n = val(rs, "records", "dispatch")
        if not n:
            continue
        light_n += n
        light["(all)"] += ms
        for r in rs:
            if r["kind"] == "phase.dispatch":
                light[r["name"]] += fnum(r["ms"])
    if light_n and disp_n_heavy:
        names = sorted(set(light) | set(disp_phase_heavy), key=lambda k: -disp_phase_heavy[k])
        print(f"dispatch phases, us per dispatch: heavy submissions ({disp_n_heavy}) vs light ones ({light_n})")
        for k in names:
            print(f"  {k:12s} {disp_phase_heavy[k] / disp_n_heavy * 1e3:8.1f} {light[k] / light_n * 1e3:8.1f}")


if __name__ == "__main__":
    main()
