"""The fp_* samples of a perf recording of nxrt by instruction kind (README "Which guest code a thread runs").

Usage: tools/venv/bin/python -I tools/perf_fp_kinds.py <perf script output> <nm symbols> <readelf -lW output>
"""
import sys, re, bisect, collections, capstone
script, syms_path, segs_path = sys.argv[1:4]
addrs, names = [], []
for line in open(syms_path):
    p = line.split()
    if len(p) >= 3: addrs.append(int(p[0], 16)); names.append(p[2])
segs = []
for line in open(segs_path):
    m = re.match(r"\s*LOAD\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+0x[0-9a-f]+\s+(0x[0-9a-f]+)\s+0x[0-9a-f]+\s+(.*)", line)
    if m and "E" in m.group(4): segs.append((int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)))
maps = []; total = 0; fp = collections.Counter()
for line in open(script, errors="replace"):
    if "PERF_RECORD_MMAP" in line:
        m = re.search(r"\[(0x[0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+)", line)
        if m and "/nxrt" in line: s = int(m.group(1), 16); maps.append((s, s + int(m.group(2), 16), int(m.group(3), 16)))
        continue
    m = re.match(r"\s*([0-9a-f]+)\s", line)
    if not m: continue
    total += 1; ip = int(m.group(1), 16)
    for s, e, pg in maps:
        if s <= ip < e:
            off = ip - s + pg
            for so, va, sz in segs:
                if so <= off < so + sz:
                    i = bisect.bisect_right(addrs, off - so + va) - 1
                    if names[i].startswith("fp_"): fp[names[i]] += 1
md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
kinds = collections.Counter(); ex = {}
for n, c in fp.items():
    w = int(n[3:], 16)
    ins = list(md.disasm(w.to_bytes(4, "little"), 0))
    if not ins: k = "?"
    else:
        a = ins[0]; ops = a.op_str
        arr = re.search(r"\.(16b|8b|8h|4h|4s|2s|2d|1d|s|h|d|b)\b", ops)
        scal = "vec " + arr.group(1) if arr and re.search(r"v\d+\.", ops) else ("scalar " + ops.split(",")[0][0] if ops else "")
        k = f"{a.mnemonic} ({scal})"
    kinds[k] += c; ex.setdefault(k, ins[0].mnemonic + " " + ins[0].op_str if ins else n)
fptot = sum(fp.values())
print(f"FP helper samples {fptot} of {total} ({100*fptot/total:.1f} %)")
acc = 0
for k, c in kinds.most_common(40):
    acc += c
    print(f"  {100*c/total:5.2f} %  (cum {100*acc/fptot:5.1f} % of helpers)  {k:28s} e.g. {ex[k]}")
