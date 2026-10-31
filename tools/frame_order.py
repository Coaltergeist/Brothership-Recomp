#!/usr/bin/env python3
"""Frame-order and band checks on consecutive captures (NVN_PLAN Phase 10, milestone L3).

Usage: python3 -I tools/frame_order.py <dir with frame_<n>.png> [--region y0:y1,x0:x1] [--band 16] [--quiet]

For frames whose numbers follow each other (a missing number, e.g. a dropped capture, only breaks the chain):
- repeat: identical to the previous frame (normal when the picture holds still or a video holds a frame);
- jump-back: different from the previous frame but identical to one 2..6 frames older (a stale buffer shown
  again: ARCHITECTURE 5.4, the title's sea "moving back and forth" on 2026-10-14);
- mixed: among the bands of --band rows that differ between frames n-1 and n+1, at least two equal n-1 and at
  least two equal n+1 (half old, half new: the banded opening movie of 2026-10-15).
Comparisons are exact (PNG is lossless and the GPU output deterministic), so the tool needs no numpy.
--region limits the comparison (rows y0:y1, columns x0:x1), e.g. 560:700,0:1280 for the title's sea.
Exit code 1 if there is a jump-back or a mixed frame, else 0.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "m1"))
from check_frame import decode  # noqa: E402  (the M1 test's PNG decoder: RGB rows)


def load(path, region):
    w, h, rows = decode(path)
    y0, y1, x0, x1 = region if region else (0, h, 0, w)
    return [bytes(r[x0 * 3:x1 * 3]) for r in rows[y0:min(y1, h)]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--region", help="y0:y1,x0:x1")
    ap.add_argument("--band", type=int, default=16)
    ap.add_argument("--quiet", action="store_true", help="only the summary")
    a = ap.parse_args()
    region = None
    if a.region:
        (y0, y1), (x0, x1) = [map(int, p.split(":")) for p in a.region.split(",")]
        region = (y0, y1, x0, x1)

    files = {}
    for f in os.listdir(a.dir):
        m = re.fullmatch(r"frame_(\d+)\.png", f)
        if m:
            files[int(m.group(1))] = os.path.join(a.dir, f)
    if not files:
        sys.exit(f"no frame_<n>.png in {a.dir}")
    nums = sorted(files)
    frames = {n: load(files[n], region) for n in nums}
    rows = len(frames[nums[0]])
    bands = lambda img: [b"".join(img[y:y + a.band]) for y in range(0, rows, a.band)]
    banded = {n: bands(frames[n]) for n in nums}

    gaps = sum(1 for p, n in zip(nums, nums[1:]) if n != p + 1)
    repeats, jumps, mixed, compared = [], [], [], 0
    for n in nums:
        if n - 1 not in frames:
            continue
        compared += 1
        if frames[n] == frames[n - 1]:
            repeats.append(n)
            continue
        for k in range(2, 7):
            if n - k not in frames:
                break
            if frames[n] == frames[n - k]:
                jumps.append((n, k))
                break
        if n + 1 in frames:
            old = new = 0
            for p, c, q in zip(banded[n - 1], banded[n], banded[n + 1]):
                if p == q:
                    continue  # this band does not change across the three frames
                if c == p:
                    old += 1
                elif c == q:
                    new += 1
            if old >= 2 and new >= 2:
                mixed.append((n, old, new))

    print(f"{len(nums)} frames ({nums[0]}..{nums[-1]}, {gaps} gap(s)), {compared} compared with their previous frame: "
          f"{len(repeats)} repeats, {len(jumps)} jump-backs, {len(mixed)} mixed")
    if not a.quiet:
        if jumps:
            print("jump-backs (frame, same as the frame this many before):", jumps[:20])
        if mixed:
            print("mixed (frame, bands from the previous, bands from the next):", mixed[:20])
    sys.exit(1 if jumps or mixed else 0)


if __name__ == "__main__":
    main()
