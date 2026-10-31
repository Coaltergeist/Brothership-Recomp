"""Summarises the metadata an uncompressed NSO carries for function discovery.

Usage: nso_info.py <module.bin> [--dump-imports FILE] [--dump-functions FILE]

Reports the MOD0 header, .dynamic tags, dynamic symbols (exports/imports),
relocations by type, .eh_frame_hdr FDE coverage, and how FDE starts compare
with direct BL targets.
"""
import argparse
import collections
import struct
from pathlib import Path

import numpy as np

DT_NAMES = {0: "NULL", 1: "NEEDED", 2: "PLTRELSZ", 3: "PLTGOT", 4: "HASH", 5: "STRTAB",
            6: "SYMTAB", 7: "RELA", 8: "RELASZ", 9: "RELAENT", 10: "STRSZ", 11: "SYMENT",
            12: "INIT", 13: "FINI", 14: "SONAME", 16: "SYMBOLIC", 20: "PLTREL", 23: "JMPREL",
            24: "BIND_NOW", 25: "INIT_ARRAY", 26: "FINI_ARRAY", 27: "INIT_ARRAYSZ",
            28: "FINI_ARRAYSZ", 30: "FLAGS", 0x6FFFFEF5: "GNU_HASH", 0x6FFFFFF9: "RELACOUNT",
            0x6FFFFFFB: "FLAGS_1"}
R_NAMES = {257: "ABS64", 1025: "GLOB_DAT", 1026: "JUMP_SLOT", 1027: "RELATIVE"}
STT = {0: "NOTYPE", 1: "OBJECT", 2: "FUNC"}


class Image:
    def __init__(self, path):
        data = Path(path).read_bytes()
        if data[:4] != b"NSO0" or struct.unpack_from("<I", data, 0xC)[0] & 7:
            raise SystemExit(f"{path}: expected an uncompressed NSO0")
        self.segs = [struct.unpack_from("<III", data, off) for off in (0x10, 0x20, 0x30)]
        bss = struct.unpack_from("<I", data, 0x3C)[0]
        end = self.segs[2][1] + self.segs[2][2] + bss
        self.mem = bytearray(end)
        for file_off, mem_off, size in self.segs:
            self.mem[mem_off:mem_off + size] = data[file_off:file_off + size]
        self.text = (self.segs[0][1], self.segs[0][1] + self.segs[0][2])

    def u32(self, a): return struct.unpack_from("<I", self.mem, a)[0]
    def s32(self, a): return struct.unpack_from("<i", self.mem, a)[0]
    def u64(self, a): return struct.unpack_from("<Q", self.mem, a)[0]

    def cstr(self, a):
        return self.mem[a:self.mem.index(0, a)].decode("utf-8", "replace")


def read_encoded(img, a, enc, pc_base=None):
    """DWARF pointer encodings used by .eh_frame_hdr (sdata4/udata4/udata8, pcrel/datarel)."""
    fmt = enc & 0x0F
    if fmt == 0x0B:
        v, n = img.s32(a), 4
    elif fmt == 0x03:
        v, n = img.u32(a), 4
    elif fmt in (0x04, 0x0C):
        v, n = img.u64(a), 8
    else:
        raise ValueError(f"unsupported pointer encoding 0x{enc:x}")
    app = enc & 0x70
    if app == 0x10:
        v += a
    elif app == 0x30:
        v += pc_base
    return v & (2**64 - 1), n


def parse(path):
    img = Image(path)
    mod0 = img.u32(4)
    if img.mem[mod0:mod0 + 4] != b"MOD0":
        raise SystemExit("no MOD0 header")
    dyn_off, bss_s, bss_e, ehf_s, ehf_e, modobj = (img.s32(mod0 + 4 + 4 * i) + mod0 for i in range(6))
    info = {"mod0": mod0, "dynamic": dyn_off, "eh_frame_hdr": (ehf_s, ehf_e), "bss": (bss_s, bss_e)}

    dyn = collections.defaultdict(list)
    a = dyn_off
    while True:
        tag, val = img.u64(a), img.u64(a + 8)
        if tag == 0:
            break
        dyn[tag].append(val)
        a += 16
    info["dyn"] = dyn

    # Symbol count from DT_HASH nchain, else the gap between SYMTAB and STRTAB.
    symtab, strtab = dyn[6][0], dyn[5][0]
    nsyms = img.u32(dyn[4][0] + 4) if 4 in dyn else (strtab - symtab) // 24
    syms = []
    for i in range(nsyms):
        s = symtab + 24 * i
        name_off, st_info, _, shndx = struct.unpack_from("<IBBH", img.mem, s)
        value, size = img.u64(s + 8), img.u64(s + 16)
        syms.append((img.cstr(strtab + name_off), STT.get(st_info & 0xF, st_info & 0xF),
                     st_info >> 4, shndx, value, size))
    info["syms"] = syms

    relocs = collections.Counter()
    reloc_targets = []
    for start_tag, size_tag in ((7, 8), (23, 2)):
        if start_tag in dyn:
            start, size = dyn[start_tag][0], dyn[size_tag][0]
            for r in range(start, start + size, 24):
                off, rinfo, addend = struct.unpack_from("<QQq", img.mem, r)
                rtype, rsym = rinfo & 0xFFFFFFFF, rinfo >> 32
                relocs[R_NAMES.get(rtype, rtype)] += 1
                if rtype == 1027:
                    reloc_targets.append(addend)
    info["relocs"] = relocs
    info["relative_targets"] = reloc_targets

    fdes = []
    if ehf_e > ehf_s:
        a = ehf_s
        version, ptr_enc, cnt_enc, tbl_enc = img.mem[a:a + 4]
        _, n = read_encoded(img, a + 4, ptr_enc)
        count, m = read_encoded(img, a + 4 + n, cnt_enc)
        t = a + 4 + n + m
        width = 8 if (tbl_enc & 0x0F) == 0x0B else 16
        for i in range(count):
            loc, _ = read_encoded(img, t + i * width, tbl_enc, pc_base=ehf_s)
            fdes.append(loc)
    info["fdes"] = fdes
    info["img"] = img
    return info


def bl_targets(img):
    lo, hi = img.text
    words = np.frombuffer(bytes(img.mem[lo:hi]), "<u4")
    idx = np.nonzero((words & 0xFC000000) == 0x94000000)[0]
    imm = (words[idx].astype(np.int64) & 0x03FFFFFF)
    imm = np.where(imm & 0x02000000, imm - 0x04000000, imm)
    targets = lo + idx.astype(np.int64) * 4 + imm * 4
    return set(int(t) for t in targets if lo <= t < hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--dump-imports")
    ap.add_argument("--dump-functions")
    args = ap.parse_args()
    info = parse(args.module)
    img = info["img"]
    lo, hi = img.text
    dyn = info["dyn"]

    print(f"module        {args.module}")
    print(f"MOD0          0x{info['mod0']:x}   .dynamic 0x{info['dynamic']:x}   "
          f".eh_frame_hdr 0x{info['eh_frame_hdr'][0]:x}-0x{info['eh_frame_hdr'][1]:x}")
    print("dynamic tags  " + ", ".join(f"{DT_NAMES.get(t, hex(t))}" + (f"x{len(v)}" if len(v) > 1 else "")
                                       for t, v in dyn.items()))
    needed = [img.cstr(dyn[5][0] + v) for v in dyn.get(1, [])]
    if needed:
        print(f"NEEDED        {needed}")

    syms = info["syms"]
    defined = [s for s in syms if s[3] != 0 and s[0]]
    imports = [s for s in syms if s[3] == 0 and s[0]]
    funcs_def = [s for s in defined if s[1] == "FUNC"]
    print(f"dynsym        {len(syms)} entries: {len(defined)} defined ({len(funcs_def)} FUNC), "
          f"{len(imports)} imported")
    print("relocations   " + ", ".join(f"{k} {v}" for k, v in info["relocs"].most_common()))

    fdes = set(info["fdes"])
    fde_text = {f for f in fdes if lo <= f < hi}
    bls = bl_targets(img)
    rel_text = {t for t in info["relative_targets"] if lo <= t < hi}
    exports = {s[4] for s in funcs_def}
    print(f"FDEs          {len(fdes)} ({len(fde_text)} in .text)")
    print(f"BL targets    {len(bls)} distinct")
    print(f"  of which are FDE starts      {len(bls & fde_text)} ({100 * len(bls & fde_text) / max(1, len(bls)):.1f}%)")
    print(f"FDE starts never BL-called     {len(fde_text - bls)}")
    print(f"RELATIVE relocs into .text     {len(rel_text)} distinct (vtables, function pointers)")
    print(f"  of which are FDE starts      {len(rel_text & fde_text)}")
    print(f"  not FDE, not BL target       {len(rel_text - fde_text - bls)}")
    union = fde_text | bls | rel_text | exports
    print(f"union of function-start evidence: {len(union)}")

    if args.dump_imports:
        with open(args.dump_imports, "w") as f:
            for s in sorted(imports):
                f.write(f"{s[1]}\t{s[0]}\n")
    if args.dump_functions:
        with open(args.dump_functions, "w") as f:
            for a in sorted(union):
                tags = "".join(c for c, sset in (("F", fde_text), ("B", bls), ("R", rel_text), ("E", exports)) if a in sset)
                f.write(f"{a:x}\t{tags}\n")


if __name__ == "__main__":
    main()
