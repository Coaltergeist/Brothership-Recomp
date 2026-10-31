#!/usr/bin/env python3
"""Checks whether a lifted function can be overridden (runtime/src/hle.h).

An override catches calls that reach a function through a pointer (rt_jump) and
PLT calls (its GOT slots are redirected), but not the direct calls between
lifted functions. This lists, for a symbols.txt name, every place the generated
C calls the function directly, by kind, and every GOT slot bound to it:

  BL            a direct call                                   bypasses an override
  B             a tail branch (conditional or not)              bypasses an override
  fall-through  the previous function runs on into this one     bypasses an override
  PLT           a guarded PLT call (`if (x17 == target)`)       caught: its slot is redirected

The generated C is the ground truth of what the lift calls directly (a scan of
the machine code for B/BL misses fall-through).

Usage: tools/direct_callers.py <gen dir> <symbol>
       e.g. tools/direct_callers.py build/m3/gen nvnBootstrapLoader
Exit status: 0 safe to override, 1 direct calls found, 2 usage error.
"""
import glob
import re
import subprocess
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    gen, symbol = sys.argv[1], sys.argv[2]
    targets = []
    with open(f"{gen}/symbols.txt") as f:
        for line in f:
            off, _, name = line.rstrip("\n").partition(" ")
            if name == symbol:
                targets.append(int(off, 16))
    if len(targets) != 1:
        print(f"{symbol}: {len(targets)} symbols.txt entries (an override needs exactly one)", file=sys.stderr)
        return 2
    target = targets[0]
    modules = []  # (name, offset, size), program-relative
    with open(f"{gen}/link.txt") as f:
        for line in f:
            m = re.match(r"\s+(\S+) 0x([0-9a-f]+) size 0x([0-9a-f]+)", line)
            if m:
                modules.append((m.group(1), int(m.group(2), 16), int(m.group(3), 16)))

    def module_of(off):
        return next((n for n, b, s in modules if b <= off < b + s), "?")

    print(f"{symbol} at 0x{target:x} (program-relative), module {module_of(target)}")
    call = f"f_{target:x}(c)"
    files = sorted(glob.glob(f"{gen}/funcs_*.c"))
    out = subprocess.run(["grep", "-F", "-n", call, *files], capture_output=True, text=True).stdout
    hits = {}  # kind -> [(file, line)]
    for row in out.splitlines():
        path, lineno, text = row.split(":", 2)
        s = text.strip()
        if "c->x[17] ==" in s:
            kind = "PLT"
        elif s == f"{call};":
            kind = "BL"
        elif s == f"{call}; return;":
            kind = "fall-through"
        else:
            kind = "B"
        hits.setdefault(kind, []).append((path, int(lineno)))

    def caller_pc(path, lineno):
        # The nearest instruction comment above the call: "/* <pc>: <insn> */".
        with open(path) as f:
            lines = f.readlines()[max(0, lineno - 40):lineno]
        for line in reversed(lines):
            m = re.search(r"/\* ([0-9a-f]+): [0-9a-f]{8} \*/", line)
            if m:
                return int(m.group(1), 16)
        return None

    bypass = 0
    for kind in ("BL", "B", "fall-through", "PLT"):
        for path, lineno in hits.get(kind, [])[:50]:
            pc = caller_pc(path, lineno)
            where = f"0x{pc:x} [{module_of(pc)}]" if pc is not None else "?"
            print(f"  {kind:12} from {where} ({path.split('/')[-1]}:{lineno})")
        if len(hits.get(kind, [])) > 50:
            print(f"  ... {len(hits[kind]) - 50} more {kind}")
        if kind != "PLT":
            bypass += len(hits.get(kind, []))
    slots = []
    try:
        with open(f"{gen}/slots.txt") as f:
            for line in f:
                slot, tgt = (int(x, 16) for x in line.split())
                if tgt == target:
                    slots.append(slot)
    except FileNotFoundError:
        print("  (no slots.txt: run nxrecomp <modules> -o <gen dir> --link-only to write it)")
    for slot in slots:
        print(f"  GOT slot 0x{slot:x} [{module_of(slot)}]: redirected by the override")
    print(f"{bypass} direct calls, {len(hits.get('PLT', []))} guarded PLT calls, {len(slots)} GOT slots: "
          + ("safe to override" if not bypass else "NOT safe: the direct calls bypass an override"))
    return 1 if bypass else 0


if __name__ == "__main__":
    sys.exit(main())
