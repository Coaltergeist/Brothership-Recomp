"""Differential tester: drippu's AArch64 -> C translation vs Unicorn (Cortex-A57).

Samples real instruction encodings from an uncompressed NSO's .text, translates
each with drippu's Translate() (via gen_tests), and runs every encoding on
identical random CPU states in both the generated C and Unicorn. Compares
X0-X30, SP, PC, NZCV, V0-V31, FPSR and a scratch memory window.

Usage (from the repo root):
  tools/venv/bin/python -I tools/difftest/difftest.py extracted/exefs/main.bin

Full mode: every distinct encoding of the chosen families in several modules,
tested once each, with integer boundary values in the vector lanes, built with the generated code's
compiler and flags, optionally through the lifter's SIMD emitter:
  tools/venv/bin/python -I tools/difftest/difftest.py --full --families int,move --simd --jobs 14 \
      extracted/exefs/{main,subsdk0,sdk}.bin

The tests are compiled as the runtime compiles the generated code (--cc, default tools/llvm/bin/clang;
--cflags, default -O1 -msse4.2). --mutate-qc removes every write of FPSR.QC from the generated tests:
a run that still passes did not exercise saturation (the check that the boundary inputs work).

Outputs go to analysis/difftest_<name>.{txt,json}; build files to build/difftest/.
"""
import argparse
import collections
import ctypes
import json
import multiprocessing
import os
import random
import re
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import capstone
import numpy as np
from capstone import arm64_const as cs64
import unicorn
from unicorn import arm64_const as uc64

REPO = Path(__file__).resolve().parents[2]
DRIPPU_SRC = REPO / "third_party" / "drippu" / "src"
LIFTER_SRC = REPO / "lifter" / "src"
DEFAULT_CC = REPO / "tools" / "llvm" / "bin" / "clang"
# drippu's two spellings, and the SIMD emitter's
# nx_mem.h's host pointer (lifter/src/fastmem.cpp: the inline fast path's, or --direct-mem's base);
# --mutate-mem moves it by one byte
MEM_FAST_PTR = re.compile(r"(\(unsigned char\*\)\(p \+ \(uintptr_t\)va\)|(?<=#define NX_MEM_BASE )0x400000000000ULL)")
QC_WRITE = re.compile(r"c->fpsr\|=(?:UINT64_C\(1\)|1ULL)<<27|c->fpsr \|= nxv_qc\([^;]*\)")
TOOL_DIR = Path(__file__).resolve().parent

SCRATCH_BASE = 0x0800_0000
SCRATCH_SIZE = 0x2_0000
SCRATCH_PTR = SCRATCH_BASE + 0x8000  # memory operands' base registers point here

# Not testable as single instructions against a bare CPU: exceptions, system
# registers/maintenance, barriers and the exclusive monitor.
EXCLUDED = {
    ".byte", "svc", "hvc", "smc", "brk", "hlt", "udf", "dcps1", "dcps2", "dcps3", "eret",
    "drps", "mrs", "msr", "sys", "sysl", "dc", "ic", "at", "tlbi", "isb", "dsb", "dmb",
    "ssbb", "pssbb", "sb", "clrex", "wfe", "wfi", "sev", "sevl", "yield", "hint",
    "ldxr", "ldxrb", "ldxrh", "ldxp", "ldaxr", "ldaxrb", "ldaxrh", "ldaxp",
    "stxr", "stxrb", "stxrh", "stxp", "stlxr", "stlxrb", "stlxrh", "stlxp",
}
SVE_OPERAND = re.compile(r"\b[zp]\d+\b")

F32_SPECIAL = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3F000000, 0x7F800000,
               0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00001, 0x00000001, 0x807FFFFF,
               0x7F7FFFFF, 0x00800000, 0x4B800001, 0x3FC00000, 0x40400000, 0x4F000000,
               0xCF000000, 0x5F000000]
F64_SPECIAL = [0x0000000000000000, 0x8000000000000000, 0x3FF0000000000000,
               0xBFF0000000000000, 0x3FE0000000000000, 0x7FF0000000000000,
               0xFFF0000000000000, 0x7FF8000000000000, 0x7FF0000000000001,
               0x0000000000000001, 0x800FFFFFFFFFFFFF, 0x7FEFFFFFFFFFFFFF,
               0x0010000000000000, 0x43E0000000000000, 0xC3E0000000000000,
               0x41E0000000000000, 0x4330000000000001]
GPR_SPECIAL = [0, 1, 2, 3, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000,
               0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x100000000, 0xFF, 0xFFFF, 0x8000, 63, 64]


class DtState(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint64 * 32), ("pc", ctypes.c_uint64),
                ("nzcv", ctypes.c_uint64), ("v", (ctypes.c_uint64 * 2) * 32),
                ("fpcr", ctypes.c_uint64), ("fpsr", ctypes.c_uint64)]


def load_nso(path):
    data = Path(path).read_bytes()
    if data[:4] != b"NSO0" or struct.unpack_from("<I", data, 0xC)[0] & 7:
        raise SystemExit(f"{path}: expected an uncompressed NSO0")
    segs = [struct.unpack_from("<III", data, off) for off in (0x10, 0x20, 0x30)]
    bss = struct.unpack_from("<I", data, 0x3C)[0]
    image_end = (segs[2][1] + segs[2][2] + bss + 0xFFF) & ~0xFFF
    if image_end > SCRATCH_BASE:
        raise SystemExit("image overlaps the scratch window")
    mem = np.zeros(SCRATCH_BASE + SCRATCH_SIZE, dtype=np.uint8)
    for file_off, mem_off, size in segs:
        mem[mem_off:mem_off + size] = np.frombuffer(data, np.uint8, size, file_off)
    text_off, text_va, text_size = segs[0]
    words = np.frombuffer(data, "<u4", text_size // 4, text_off)
    return mem, image_end, words, text_va


FP_EXTRA = {"scvtf", "ucvtf", "bfcvt", "bfcvtn", "bfcvtn2", "bfdot", "bfmmla", "bfmlalb", "bfmlalt"}
MOVE = {"mov", "movi", "mvni", "dup", "ins", "umov", "smov", "orr", "orn", "and", "eor", "bic", "bsl", "bit",
        "bif", "ext", "zip1", "zip2", "uzp1", "uzp2", "trn1", "trn2", "tbl", "tbx", "rev16", "rev32", "rev64",
        "not", "mvn"}


def family(enc, mnem):
    """FP/SIMD data processing (op0 = x111) split into classes: fp, move (moves, permutes,
    bitwise logic), int (the rest of integer SIMD); loads and stores (op0 = x1x0) are "mem"; everything
    else is "other"."""
    if (enc >> 25) & 5 == 4:
        return "mem"
    if (enc >> 25) & 7 != 7:
        return "other"
    if mnem.startswith("f") or mnem in FP_EXTRA:
        return "fp"
    return "move" if mnem in MOVE else "int"


def sample_encodings(words, text_va, per_mnemonic, rng, families=None, full=False, seen=None):
    uniq, first, counts = np.unique(words, return_index=True, return_counts=True)
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    md.skipdata = True
    groups = collections.defaultdict(list)
    sites = collections.Counter()
    for (addr, size, mnem, ops), enc, idx, cnt in zip(
            md.disasm_lite(uniq.astype("<u4").tobytes(), 0), uniq, first, counts):
        if mnem in EXCLUDED or SVE_OPERAND.search(ops):
            continue
        if families and family(int(enc), mnem) not in families:
            continue
        sites[mnem] += int(cnt)
        if seen is not None and int(enc) in seen:  # tested from an earlier module
            continue
        groups[mnem].append((int(cnt), int(enc), text_va + int(idx) * 4))
    chosen = []
    for mnem, encs in groups.items():
        if full:
            chosen += [(mnem, enc, pc, cnt) for cnt, enc, pc in sorted(encs, reverse=True)]
            continue
        encs.sort(reverse=True)
        top = encs[:per_mnemonic // 2]
        rest = encs[per_mnemonic // 2:]
        pick = top + rng.sample(rest, min(len(rest), per_mnemonic - len(top)))
        chosen += [(mnem, enc, pc, cnt) for cnt, enc, pc in pick]
    return chosen, sites


def build_library(cases, build_dir, fpx=False, simd=False, cc=None, cflags=None, mutate_qc=False, inline_mem=False,
                  mutate_mem=False, jobs=None, direct_mem=False, mutate_fpx=False):
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "encodings.txt").write_text("".join(f"{e:08x} {pc:x}\n" for _, e, pc, _ in cases))
    for old in build_dir.glob("tests_*.c"):
        old.unlink()
    gen = build_dir / "gen_tests"
    subprocess.run(["g++", "-std=c++20", "-O1", f"-I{DRIPPU_SRC}", f"-I{LIFTER_SRC}", str(TOOL_DIR / "gen_tests.cpp"),
                    str(LIFTER_SRC / "simd.cpp"), str(LIFTER_SRC / "fastmem.cpp"), "-o", str(gen)], check=True)
    subprocess.run([str(gen), str(build_dir / "encodings.txt"), str(build_dir)] + (["--fpx"] if fpx else []) +
                   (["--simd"] if simd else []) + (["--inline-mem"] if inline_mem else []) +
                   (["--direct-mem"] if direct_mem else []), check=True)
    parts = sorted(build_dir.glob("tests_*.c"))
    if mutate_qc:
        n = 0
        for tests in parts:
            text, k = QC_WRITE.subn("(void)0", tests.read_text())
            tests.write_text(text)
            n += k
        print(f"--mutate-qc: removed {n} FPSR.QC writes from the tests")
    if mutate_mem:
        hdr = build_dir / "nx_mem.h"
        text, k = MEM_FAST_PTR.subn(r"(\1 + 1)", hdr.read_text())
        if k != 1:
            sys.exit("--mutate-mem: nx_mem.h's fast-path pointer not found")
        hdr.write_text(text)
        print("--mutate-mem: every inline / direct access moved by one byte")
    if mutate_fpx:
        hdr = build_dir / "nx_simd.h"
        text, k = re.subn(r"#define NXV_FPX_MUTATE\n", "#define NXV_FPX_MUTATE ^ 1u\n", hdr.read_text())
        if k != 1:
            sys.exit("--mutate-fpx: nx_simd.h's NXV_FPX_MUTATE not found")
        hdr.write_text(text)
        print("--mutate-fpx: every result the FP fast path keeps has its low bit flipped")
    lib = build_dir / "difftest.so"
    probe = ["-include", str(TOOL_DIR / "fpx_probe.h")] if fpx else []
    sources = parts + [build_dir / "tests.c", build_dir / "recomp_runtime.c", TOOL_DIR / "harness.c"]
    compile_ = lambda src: subprocess.run([cc, *cflags, "-fPIC", "-c", "-w", f"-I{build_dir}"] + probe +
                                          [str(src), "-o", str(build_dir / (src.stem + ".o"))], check=True)
    with ThreadPoolExecutor(jobs or os.cpu_count()) as pool:  # one compiler process per file
        list(pool.map(compile_, sources))
    subprocess.run([cc, "-shared", "-o", str(lib)] + [str(build_dir / (s.stem + ".o")) for s in sources] +
                   ["-lm"], check=True)
    return lib


def load_library(lib):
    dll = ctypes.CDLL(str(lib))
    dll.dt_fpx_probe.restype = ctypes.c_ulonglong
    dll.dt_run.argtypes = [ctypes.c_int, ctypes.POINTER(DtState), ctypes.c_void_p,
                           ctypes.c_uint64, ctypes.c_uint64]
    return dll


def reg_index(md, reg):
    name = md.reg_name(reg)
    if name in ("sp", "wsp"):
        return 31
    if name == "fp":
        return 29
    if name == "lr":
        return 30
    if name and name[0] in "xw" and name[1:].isdigit():
        return int(name[1:])
    return None


INDIRECT_BRANCHES = {"br", "blr", "ret"}


def operand_regs(md, enc):
    """Base/index registers of memory operands, and the target of an indirect branch."""
    insn = next(md.disasm(struct.pack("<I", enc), 0), None)
    bases, indexes, targets = set(), set(), set()
    if insn is None:
        return bases, indexes, targets
    if insn.mnemonic in INDIRECT_BRANCHES:
        regs = [reg_index(md, op.reg) for op in insn.operands if op.type == cs64.ARM64_OP_REG]
        targets.add(regs[0] if regs else 30)
    for op in insn.operands:
        if op.type == cs64.ARM64_OP_MEM:
            if op.mem.base and (b := reg_index(md, op.mem.base)) is not None:
                bases.add(b)
            if op.mem.index and (i := reg_index(md, op.mem.index)) is not None:
                indexes.add(i)
    return bases, indexes, targets


def random_gpr(rng):
    r = rng.random()
    if r < 0.4:
        return rng.getrandbits(64)
    if r < 0.6:
        return rng.randrange(65)
    return rng.choice(GPR_SPECIAL)


def random_half(rng):
    r = rng.random()
    if r < 0.3:
        return rng.getrandbits(64)
    if r < 0.65:
        lo = rng.choice(F32_SPECIAL) if rng.random() < 0.7 else struct.unpack(
            "<I", struct.pack("<f", rng.uniform(-1e4, 1e4)))[0]
        hi = rng.choice(F32_SPECIAL) if rng.random() < 0.7 else struct.unpack(
            "<I", struct.pack("<f", rng.uniform(-1e4, 1e4)))[0]
        return lo | (hi << 32)
    if rng.random() < 0.7:
        return rng.choice(F64_SPECIAL)
    return struct.unpack("<Q", struct.pack("<d", rng.uniform(-1e6, 1e6)))[0]


def int_boundaries(bits):
    """Lane values where integer SIMD changes behaviour: 0, +-1, the type's limits and one off, and the
    same around the half-width limits that narrowing (and saturating narrowing) instructions clamp to."""
    m = (1 << bits) - 1
    h = bits // 2
    vals = {0, 1, 2, m, m - 1, 1 << (bits - 1), (1 << (bits - 1)) - 1, (1 << (bits - 1)) + 1}
    if h:
        for v in ((1 << h) - 1, 1 << h, (1 << h) + 1, (1 << (h - 1)) - 1, 1 << (h - 1),
                  -(1 << (h - 1)), -(1 << (h - 1)) - 1, -(1 << h), -(1 << h) + 1):
            vals.add(v & m)
    return sorted(vals)


INT_BOUNDARIES = {b: int_boundaries(b) for b in (8, 16, 32, 64)}


def random_int_vector(rng):
    """A 128-bit register as (lo, hi): lanes of one random size, each a boundary value or random;
    sometimes all lanes the same value (so two registers can meet at, e.g., both minimal)."""
    bits = rng.choice((8, 16, 32, 64))
    lanes = 128 // bits
    if rng.random() < 0.15:
        same = rng.choice(INT_BOUNDARIES[bits])
        vals = [same] * lanes
    else:
        vals = [rng.choice(INT_BOUNDARIES[bits]) if rng.random() < 0.75 else rng.getrandbits(bits)
                for _ in range(lanes)]
    q = 0
    for i, v in enumerate(vals):
        q |= v << (i * bits)
    return q & (2**64 - 1), q >> 64


def extreme_values(bits):
    m = (1 << bits) - 1
    return [1 << (bits - 1), (1 << (bits - 1)) - 1, 0, m, (1 << (bits - 1)) + 1]  # min, max, 0, -1, min + 1


LANE_BITS = {"b": 8, "h": 16, "s": 32, "d": 64}


def lane_bits(op_str):
    """Lane sizes the instruction's arrangements name (".8h" -> 16), or none."""
    return sorted({LANE_BITS[m] for m in re.findall(r"\.\d*([bhsd])\b", op_str)})


def random_state(rng, pc, bases, indexes, targets, text_range, integer=False, hint=()):
    s = DtState()
    for r in range(32):
        s.x[r] = random_gpr(rng)
    for r in bases:
        s.x[r] = SCRATCH_PTR + rng.randrange(256) * 16
    for r in indexes - bases:
        s.x[r] = rng.randrange(256)
    for r in targets:  # Unicorn faults on a branch to unmapped memory
        s.x[r] = rng.randrange(*text_range) & ~3
    s.pc = pc
    s.nzcv = rng.randrange(16) << 28
    mode = rng.random() if integer else 1.0
    bits = rng.choice(hint) if hint and rng.random() < 0.75 else rng.choice((8, 16, 32, 64))
    for r in range(32):
        if mode < 0.35:  # every register, every lane at an extreme of one lane size: both operands meet there
            q = 0
            for i in range(128 // bits):
                q |= rng.choice(extreme_values(bits)) << (i * bits)
            s.v[r][0], s.v[r][1] = q & (2**64 - 1), q >> 64
        elif mode < 0.8:
            s.v[r][0], s.v[r][1] = random_int_vector(rng)
        else:
            s.v[r][0] = random_half(rng)
            s.v[r][1] = random_half(rng)
    if not integer and rng.random() < 0.25:
        # A running game's state: ordinary normal floats (now and then a zero) in every lane
        # of the instruction's lane size, FPCR default, FPSR.IXC set, so FPX fast paths (drippu's and the
        # lifter's vectorised ones) keep results on all lanes; the other trials keep the special values.
        fbits = 64 if 64 in hint else 32
        for r in range(32):
            lanes = []
            for _ in range(128 // fbits):
                v = 0.0 if rng.random() < 0.08 else rng.choice((-1, 1)) * 10 ** rng.uniform(-6, 6)
                lanes.append(struct.unpack("<Q", struct.pack("<d", v))[0] if fbits == 64
                             else struct.unpack("<I", struct.pack("<f", v))[0])
            q = 0
            for i, lane in enumerate(lanes):
                q |= lane << (i * fbits)
            s.v[r][0], s.v[r][1] = q & (2**64 - 1), q >> 64
        s.fpcr = 0
        s.fpsr = 0x10 | rng.choice((0, 0, 0x1, 0x8))
        return s
    if rng.random() < 0.6:
        s.fpcr = 0
    else:  # RMode, FZ, DN
        s.fpcr = (rng.randrange(4) << 22) | (rng.randrange(2) << 24) | (rng.randrange(2) << 25)
    # Sticky exception flags. IXC (bit 4) matters most: drippu's FPX fast paths
    # only engage once the guest's FPSR already has it set, as in a running game.
    s.fpsr = (0x10 if rng.random() < 0.5 else 0) | (rng.choice((0, 0, 0, 0x1, 0x8, 0x80, 0x9F)))
    return s


X_REGS = [getattr(uc64, f"UC_ARM64_REG_X{i}") for i in range(31)]
Q_REGS = [getattr(uc64, f"UC_ARM64_REG_Q{i}") for i in range(32)]


def make_unicorn(mem, image_end):
    uc = unicorn.Uc(unicorn.UC_ARCH_ARM64, unicorn.UC_MODE_ARM)
    uc.ctl_set_cpu_model(uc64.UC_CPU_ARM64_A57)
    uc.mem_map(0, image_end)
    uc.mem_write(0, mem[:image_end].tobytes())
    uc.mem_map(SCRATCH_BASE, SCRATCH_SIZE)
    return uc


def run_unicorn(uc, s):
    for i in range(31):
        uc.reg_write(X_REGS[i], s.x[i])
    uc.reg_write(uc64.UC_ARM64_REG_SP, s.x[31])
    uc.reg_write(uc64.UC_ARM64_REG_NZCV, s.nzcv)
    for i in range(32):
        uc.reg_write(Q_REGS[i], s.v[i][0] | (s.v[i][1] << 64))
    uc.reg_write(uc64.UC_ARM64_REG_FPCR, s.fpcr)
    uc.reg_write(uc64.UC_ARM64_REG_FPSR, s.fpsr)
    uc.emu_start(s.pc, 0xFFFF_FFFF_FFFF_FFFF, count=1)
    out = DtState()
    for i in range(31):
        out.x[i] = uc.reg_read(X_REGS[i])
    out.x[31] = uc.reg_read(uc64.UC_ARM64_REG_SP)
    out.pc = uc.reg_read(uc64.UC_ARM64_REG_PC)
    out.nzcv = uc.reg_read(uc64.UC_ARM64_REG_NZCV) & 0xF000_0000
    for i in range(32):
        q = uc.reg_read(Q_REGS[i])
        out.v[i][0], out.v[i][1] = q & (2**64 - 1), q >> 64
    out.fpcr = uc.reg_read(uc64.UC_ARM64_REG_FPCR)
    out.fpsr = uc.reg_read(uc64.UC_ARM64_REG_FPSR)
    return out


def diff_states(ref, got, ref_mem, got_mem):
    fields = {}
    for i in range(32):
        if ref.x[i] != got.x[i]:
            fields["sp" if i == 31 else f"x{i}"] = (ref.x[i], got.x[i])
    for i in range(32):
        r = ref.v[i][0] | (ref.v[i][1] << 64)
        g = got.v[i][0] | (got.v[i][1] << 64)
        if r != g:
            fields[f"v{i}"] = (r, g)
    for name in ("pc", "nzcv", "fpsr"):
        if getattr(ref, name) != getattr(got, name):
            fields[name] = (getattr(ref, name), getattr(got, name))
    if ref_mem != got_mem:
        a = np.frombuffer(ref_mem, np.uint8)
        b = np.frombuffer(got_mem, np.uint8)
        where = np.nonzero(a != b)[0]
        fields["mem"] = (f"{len(where)} bytes differ from +0x{where[0]:x}", "")
    return fields


def snapshot(s):
    c = DtState()
    ctypes.pointer(c)[0] = s
    return c


WORKER = {}


def worker_init(lib, modules, fastmem=0, direct=False):
    WORKER["dll"] = load_library(lib)
    WORKER["dll"].dt_set_fastmem(fastmem)
    WORKER["direct"] = None  # --direct-mem: (window as a numpy array, module copied into it)
    if direct:
        WORKER["dll"].dt_direct_window.restype = ctypes.c_void_p
        w = WORKER["dll"].dt_direct_window()
        if not w:
            raise SystemExit("--direct-mem: cannot reserve the guest window at host 0x400000000000")
        n = len(modules[0][0])
        WORKER["direct"] = [np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(w)), None]
    WORKER["modules"] = modules  # (mem, image_end, words, text_va) per module, shared by fork
    WORKER["uc"] = {}
    WORKER["md"] = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    WORKER["md"].detail = True


def run_case(job):
    """Runs one encoding's trials in a worker; returns its result entry."""
    idx, mod, mnem, enc, pc, cnt, trials, seed = job
    dll, md = WORKER["dll"], WORKER["md"]
    mem, image_end, words, text_va = WORKER["modules"][mod]
    if mod not in WORKER["uc"]:
        WORKER["uc"][mod] = make_unicorn(mem, image_end)
    if WORKER["direct"]:  # the generated code reads the window at the fixed base: this module's memory there
        win = WORKER["direct"]
        if win[1] != mod:
            win[0][:] = mem
            win[1] = mod
        mem = win[0]
    uc = WORKER["uc"][mod]
    rng = random.Random(seed)
    text = next(md.disasm(struct.pack("<I", enc), pc), None)
    disasm = f"{text.mnemonic} {text.op_str}" if text else f"{enc:08x}"
    fam = family(enc, mnem)
    entry = {"mnemonic": mnem, "family": fam, "encoding": f"{enc:08x}", "pc": f"{pc:x}", "module": mod,
             "sites": cnt, "disasm": disasm, "status": "pass", "simd": bool(dll.dt_simd(idx)), "trials": 0,
             "failed": 0, "fields": collections.Counter(), "example": None}
    if dll.dt_unhandled(idx):
        entry["status"] = "unhandled"
        return entry
    bases, indexes, targets = operand_regs(md, enc)
    probe0 = (dll.dt_fpx_probe(1), dll.dt_fpx_probe(2))  # FPX hits counted in this process (--fpx)
    text_range = (text_va, text_va + len(words) * 4)
    scratch = slice(SCRATCH_BASE, SCRATCH_BASE + SCRATCH_SIZE)
    mem_ptr = mem.ctypes.data
    for _ in range(trials):
        state = random_state(rng, pc, bases, indexes, targets, text_range, integer=fam in ("int", "move"),
                             hint=lane_bits(text.op_str) if text else ())
        init = bytes(rng.getrandbits(8) for _ in range(64)) * (SCRATCH_SIZE // 64)
        uc.mem_write(SCRATCH_BASE, init)
        try:
            ref = run_unicorn(uc, state)
        except unicorn.UcError as e:
            entry["status"] = f"reference error: {e}"
            break
        ref_mem = bytes(uc.mem_read(SCRATCH_BASE, SCRATCH_SIZE))
        mem[scratch] = np.frombuffer(init, np.uint8)
        got = snapshot(state)
        halted = dll.dt_run(idx, ctypes.byref(got), mem_ptr, 0, len(mem))
        entry["trials"] += 1
        fields = {"halted": (0, halted)} if halted else diff_states(ref, got, ref_mem, mem[scratch].tobytes())
        if fields:
            entry["failed"] += 1
            entry["fields"].update(fields.keys())
            if entry["example"] is None:
                entry["example"] = {
                    "input": {"x": [f"{v:x}" for v in state.x], "nzcv": f"{state.nzcv:x}",
                              "fpcr": f"{state.fpcr:x}", "fpsr": f"{state.fpsr:x}",
                              "v": [f"{state.v[i][1]:016x}{state.v[i][0]:016x}" for i in range(32)]},
                    "diff": {k: [f"{a:x}" if isinstance(a, int) else a,
                                 f"{b:x}" if isinstance(b, int) else b]
                             for k, (a, b) in fields.items()}}
    if entry["status"] == "pass" and entry["failed"]:
        entry["status"] = "fpsr-only" if set(entry["fields"]) == {"fpsr"} else "fail"
    entry["fpx"] = [dll.dt_fpx_probe(1) - probe0[0], dll.dt_fpx_probe(2) - probe0[1]]
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modules", nargs="+")
    ap.add_argument("--per-mnemonic", type=int, default=8)
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fpx", action="store_true", help="test drippu's native-FP fast paths")
    ap.add_argument("--full", action="store_true", help="every distinct encoding, not a sample")
    ap.add_argument("--families", help="comma list of int, move, fp, mem, other (default: all)")
    ap.add_argument("--simd", action="store_true", help="through the lifter's SIMD emitter, as nxrecomp")
    ap.add_argument("--cc", default=str(DEFAULT_CC) if DEFAULT_CC.exists() else "gcc")
    ap.add_argument("--cflags", default="-O1 -msse4.2", help="as the runtime builds the generated code")
    ap.add_argument("--mutate-qc", action="store_true", help="remove FPSR.QC writes (a planted bug)")
    ap.add_argument("--inline-mem", action="store_true", help="loads and stores through nx_mem.h, as nxrecomp")
    ap.add_argument("--direct-mem", action="store_true",
                    help="loads and stores as nxrecomp --direct-mem (guest memory at a fixed host address)")
    ap.add_argument("--mutate-fpx", action="store_true",
                    help="with --simd --fpx: flip the low bit of every result the FP fast path keeps (a planted bug)")
    ap.add_argument("--mutate-mem", action="store_true",
                    help="with --inline-mem / --direct-mem: move every access by one byte (a planted bug)")
    ap.add_argument("--fastmem", type=int, choices=(0, 1, 2), default=None,
                    help="memory as a host page table: 0 none (flat), 1 all plain pages, 2 every other page "
                         "GPU-tracked (the callbacks); default 1 with --inline-mem, else 0")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--name", help="output name (default: from the first module and the options)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.mutate_fpx and not (args.simd and args.fpx):
        ap.error("--mutate-fpx needs --simd and --fpx")
    if args.mutate_mem and not (args.inline_mem or args.direct_mem):
        ap.error("--mutate-mem needs --inline-mem or --direct-mem")
    if args.inline_mem and args.direct_mem:
        ap.error("--inline-mem and --direct-mem exclude each other")
    families = set(args.families.split(",")) if args.families else None
    name = args.name or ("+".join(Path(m).name.split(".")[0] for m in args.modules) +
                         ("_full" if args.full else "") + (f"_{args.families.replace(',', '+')}" if families else "") +
                         ("_simd" if args.simd else "") + ("_fpx" if args.fpx else "") +
                         ("_mutqc" if args.mutate_qc else "") + ("_imem" if args.inline_mem else "") + ("_dmem" if args.direct_mem else "") +
                         ("_mutmem" if args.mutate_mem else "") + ("_mutfpx" if args.mutate_fpx else ""))
    fastmem = args.fastmem if args.fastmem is not None else (1 if args.inline_mem else 0)
    if fastmem:
        name += f"_fm{fastmem}" if not args.name else ""
    modules, cases, sites, seen = [], [], collections.Counter(), set()
    for mi, path in enumerate(args.modules):
        mem, image_end, words, text_va = load_nso(path)
        modules.append((mem, image_end, words, text_va))
        chosen, msites = sample_encodings(words, text_va, args.per_mnemonic, rng, families, args.full, seen)
        sites.update(msites)
        cases += [(mi, *c) for c in chosen]
        seen |= {c[1] for c in chosen}
    print(f"{name}: {len(cases)} encodings across {len(sites)} mnemonics; building with {args.cc} {args.cflags}...")
    lib = build_library([(c[1], c[2], c[3], c[4]) for c in cases], REPO / "build" / "difftest" / name, args.fpx,
                        args.simd, args.cc, args.cflags.split(), args.mutate_qc, args.inline_mem,
                        args.mutate_mem, args.jobs, args.direct_mem, args.mutate_fpx)
    if args.inline_mem and not args.mutate_mem:  # nx_mem.h's page-boundary and fallback cases (fastmem_test.c)
        build_dir = REPO / "build" / "difftest" / name
        exe = build_dir / "fastmem_test"
        subprocess.run([args.cc, *args.cflags.split(), "-w", f"-I{build_dir}", str(TOOL_DIR / "fastmem_test.c"),
                        str(build_dir / "recomp_runtime.c"), "-o", str(exe), "-lm"], check=True)
        if subprocess.run([str(exe)]).returncode != 0:
            sys.exit("fastmem_test failed")

    # Saturating instructions get more trials: their QC needs particular operand pairs.
    jobs = [(i, mi, mnem, enc, pc, cnt, args.trials * (4 if mnem.startswith(("sq", "uq")) else 1),
             rng.getrandbits(64))
            for i, (mi, mnem, enc, pc, cnt) in enumerate(cases)]
    if args.jobs > 1:
        with multiprocessing.get_context("fork").Pool(args.jobs, worker_init, (lib, modules, fastmem, args.direct_mem)) as pool:
            results = pool.map(run_case, jobs, chunksize=max(1, len(jobs) // (args.jobs * 16)))
    else:
        worker_init(lib, modules, fastmem, args.direct_mem)
        results = [run_case(j) for j in jobs]
    fpx = (sum(r.get("fpx", (0, 0))[0] for r in results), sum(r.get("fpx", (0, 0))[1] for r in results))
    write_report(name, results, sites, fpx if args.fpx else None, args)


def write_report(name, results, sites, fpx=None, args=None):
    out_dir = REPO / "analysis"
    out_dir.mkdir(exist_ok=True)
    for r in results:
        r["fields"] = dict(r["fields"])
    (out_dir / f"difftest_{name}.json").write_text(json.dumps(results, indent=1))

    status = collections.Counter(r["status"].split(":")[0] for r in results)
    by_mnem = collections.defaultdict(list)
    for r in results:
        by_mnem[r["mnemonic"]].append(r)
    total_sites = sum(sites.values())
    mode = "lifter SIMD emitter, else drippu Translate()" if args and args.simd else "drippu Translate()"
    lines = [f"difftest {name}: {mode} vs Unicorn Cortex-A57" +
             (f"; built with {Path(args.cc).name} {args.cflags}" if args else ""),
             "encodings: " + ", ".join(f"{k} {v}" for k, v in status.most_common())]
    fams = collections.defaultdict(collections.Counter)
    for r in results:
        f = fams[r.get("family", "?")]
        f["encodings"] += 1
        f["sites"] += r["sites"]
        f["simd"] += bool(r.get("simd"))
        f[r["status"].split(":")[0]] += 1
    for fam, c in sorted(fams.items()):
        lines.append(f"  {fam:<6} {c['encodings']:>6} encodings ({c['sites']} sites): {c['simd']} from the SIMD emitter; "
                     f"pass {c['pass']}, fail {c['fail']}, fpsr-only {c['fpsr-only']}, unhandled {c['unhandled']}, "
                     f"reference error {c['reference error']}")
    if fpx:
        lines.append(f"FPX fast path taken {fpx[0]} times, exact fallback {fpx[1]} times")
    census = collections.defaultdict(collections.Counter)  # the census
    for r in results:
        if r.get("family") in ("int", "move"):
            c = census[(r["family"], r["mnemonic"])]
            c["encodings"] += 1
            c["sites"] += r["sites"]
            c["simd"] += bool(r.get("simd"))
            c["simd_sites"] += r["sites"] if r.get("simd") else 0
    if census:
        lines += ["", "integer SIMD census (sites in the modules; encodings; from the SIMD emitter):"]
        for (fam, mnem), c in sorted(census.items(), key=lambda kv: -kv[1]["sites"]):
            lines.append(f"  {fam:<5}{mnem:<12}{c['sites']:>8} sites {c['encodings']:>6} encodings  "
                         f"emitter {c['simd']:>5} ({100 * c['simd_sites'] / max(1, c['sites']):5.1f}% of sites)")
    lines.append("")
    lines.append(f"{'mnemonic':<12}{'sites':>9}{'share':>8}  {'fail/tested':>11}  mismatching fields")
    ranked = sorted(by_mnem.items(), key=lambda kv: -sites[kv[0]])
    for mnem, rs in ranked:
        bad = [r for r in rs if r["status"] in ("fail", "fpsr-only")]
        if not bad:
            continue
        fields = collections.Counter()
        for r in bad:
            fields.update(r["fields"])
        tested = sum(1 for r in rs if r["trials"])
        lines.append(f"{mnem:<12}{sites[mnem]:>9}{100 * sites[mnem] / total_sites:>7.2f}%  "
                     f"{len(bad):>4}/{tested:<6}  " + ", ".join(f for f, _ in fields.most_common(6)))
    errors = [r for r in results if r["status"].startswith("reference error")]
    if errors:
        lines += ["", "reference errors (Unicorn A57 rejected; likely not ARMv8.0 or data):"]
        lines += [f"  {r['disasm']:<40} sites {r['sites']}" for r in errors]
    lines += ["", "first failing example per mnemonic (reference -> recompiled):"]
    for mnem, rs in ranked:
        for r in rs:
            if r["status"] in ("fail", "fpsr-only"):
                lines.append(f"  {r['disasm']:<44} [{r['encoding']}] {r['failed']}/{r['trials']} trials")
                for k, (a, b) in list(r["example"]["diff"].items())[:4]:
                    lines.append(f"      {k:<6} {a} -> {b}")
                break
    (out_dir / f"difftest_{name}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
