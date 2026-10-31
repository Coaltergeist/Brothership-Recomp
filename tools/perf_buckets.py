"""Self time of a perf recording of nxrt by bucket and top symbols.

Usage: python3 -I tools/perf_buckets.py <perf script output with mmap events> <nm symbols (sorted)> <readelf -lW output>
The perf script output must come from: perf script -i perf.data --show-mmap-events -F ip,sym,dso (no callchains,
-G or without -g fields): one leaf ip per sample.
"""
import bisect
import collections
import re
import sys

script, syms_path, segs_path = sys.argv[1:4]

addrs, names = [], []
for line in open(syms_path):
    parts = line.split()
    if len(parts) >= 3:
        addrs.append(int(parts[0], 16))
        names.append(parts[2])

# executable LOAD segments: (file offset, vaddr, filesz)
segs = []
for line in open(segs_path):
    m = re.match(r"\s*LOAD\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+0x[0-9a-f]+\s+(0x[0-9a-f]+)\s+0x[0-9a-f]+\s+(.*)", line)
    if m and "E" in m.group(4):
        segs.append((int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)))

maps = []  # (start, end, pgoff) of nxrt's mappings
samples = collections.Counter()
other = collections.Counter()
for line in open(script, errors="replace"):
    if "PERF_RECORD_MMAP" in line:
        m = re.search(r"\[(0x[0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+|\d+)", line)
        if m and line.rstrip().endswith("nxrt") or (m and "/nxrt" in line):
            start, size = int(m.group(1), 16), int(m.group(2), 16)
            pg = m.group(3)
            pgoff = int(pg, 16) if pg.startswith("0x") else int(pg)
            maps.append((start, start + size, pgoff))
        continue
    m = re.match(r"\s*([0-9a-f]+)\s*(.*)", line)
    if not m:
        continue
    ip = int(m.group(1), 16)
    rest = m.group(2)
    hit = None
    for start, end, pgoff in maps:
        if start <= ip < end:
            hit = ip - start + pgoff
            break
    if hit is None:
        dso = re.search(r"\(([^)]*)\)\s*$", rest)
        other[dso.group(1).split("/")[-1] if dso else "?"] += 1
        continue
    va = None
    for off, vaddr, filesz in segs:
        if off <= hit < off + filesz:
            va = hit - off + vaddr
    if va is None:
        other["nxrt (outside text)"] += 1
        continue
    i = bisect.bisect_right(addrs, va) - 1
    samples[names[i] if i >= 0 else "?"] += 1

total = sum(samples.values()) + sum(other.values())
def bucket(n):
    if n.startswith("f_"): return "lifted game code (f_*)"
    if n.startswith("fp_"): return "FP/SIMD helpers (fp_*)"
    if n.startswith("recomp_") or n.startswith("simd_"): return "drippu helpers (recomp_*)"
    return "runtime / other nxrt code"
buckets = collections.Counter()
for n, c in samples.items():
    buckets[bucket(n)] += c
for k, c in other.items():
    buckets["outside nxrt: " + k] += c
print(f"{total} samples")
for k, c in buckets.most_common():
    print(f"  {100 * c / total:5.1f} %  {k}")
print("top symbols:")
for n, c in samples.most_common(30):
    print(f"  {100 * c / total:5.2f} %  {n}")
fp = collections.Counter({n: c for n, c in samples.items() if n.startswith("fp_")})
print(f"FP helpers: {len(fp)} distinct; top 15:")
for n, c in fp.most_common(15):
    print(f"  {100 * c / total:5.2f} %  {n}")
