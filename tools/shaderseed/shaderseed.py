#!/usr/bin/env python3
"""Brothership's shader stages from the user's own game files, for the runtime's seed (SHADER_PLAN Phase 5).

The game ships every shader precompiled (Maxwell code + NVN control section) inside its assets: each stage an
Unreal 4.27 FShaderMapResourceCode entry (int32 size, LZ4 bytes, int32 unpacked size, uint8 frequency) whose
unpacked blob starts with "LSLGS?" and holds the stage's code memory (NVN debug tag 0x12345678/9, program at
+0x30 / +0x100, constants), its 0x800-byte control section (magic 0x98761234) and its NVIDIA assembly text. Material
shaders are in the IoStore container's packages, the global shaders in the .pak's GlobalShaderCache (Oodle). See
SHADER_PLAN §9.

  shaderseed.py stages <unpacked chunks dir> <pak> <out dir> [--oodle <oo2core_9_win64.dll>]

writes every distinct stage as <out>/<sha1>.ctrl / .code / .asm (RT_NVN_SEED's input). The chunks dir is
`retoc unpack <Vermillion-Switch.utoc> <dir>` (retoc: tools/retoc). Under WSL the Oodle step runs in Windows
Python (python.exe on PATH, or --windows-python). Nothing here leaves the user's PC; no key or game data is
embedded.
"""
import argparse
import hashlib
import os
import struct
import subprocess
import sys
import zlib

CONTROL_MAGIC = 0x98761234
TAG_MAGICS = (0x12345678, 0x12345679)


def lz4_block(src, size):
    """A plain LZ4 block, decoded to exactly `size` bytes; None if it does not decode so."""
    out = bytearray()
    i, n = 0, len(src)
    try:
        while i < n:
            t = src[i]
            i += 1
            l = t >> 4
            if l == 15:
                while True:
                    b = src[i]
                    i += 1
                    l += b
                    if b != 255:
                        break
            out += src[i:i + l]
            i += l
            if i >= n:
                break
            off = src[i] | src[i + 1] << 8
            i += 2
            if off == 0 or off > len(out):
                return None
            m = t & 15
            if m == 15:
                while True:
                    b = src[i]
                    i += 1
                    m += b
                    if b != 255:
                        break
            m += 4
            if off >= m:
                out += out[-off:len(out) - off + m]
            else:
                for _ in range(m):
                    out.append(out[-off])
            if len(out) > size:
                return None
    except IndexError:
        return None
    return bytes(out) if len(out) == size else None


def entries(data, seen):
    """Unpacked shader blobs of the entries in `data` not seen before (by their packed bytes' hash).

    An entry's packed bytes start with LZ4's first token, its literals with "LSLGS" (a stored entry: "LSLGS" at
    once), so each "LSLGS" is tried as the start of the literals after a token with 0-4 extra length bytes."""
    at = data.find(b"LSLGS")
    while at >= 0:
        for k in range(-1, 5):  # -1: stored, not packed
            start = at if k < 0 else at - 1 - k
            if start < 4:
                continue
            n = struct.unpack_from("<i", data, start - 4)[0]
            if not 16 <= n <= len(data) - start - 5:
                continue
            u, freq = struct.unpack_from("<iB", data, start + n)
            if freq >= 16 or not (u == n if k < 0 else n < u < 64 * n):
                continue
            packed = data[start:start + n]
            h = hashlib.sha1(packed).digest()
            if h in seen:
                break
            blob = packed if k < 0 else lz4_block(packed, u)
            if blob is not None and blob.startswith(b"LSLGS"):
                seen.add(h)
                yield blob
                break
        at = data.find(b"LSLGS", at + 5)


def split(blob):
    """(code memory, control section, assembly text) of a blob, or None."""
    tag = min((p for p in (blob.find(struct.pack("<I", m)) for m in TAG_MAGICS) if p >= 0), default=-1)
    ctl = blob.find(struct.pack("<I", CONTROL_MAGIC))
    if tag < 0 or ctl < 0 or ctl + 0x800 > len(blob):
        return None
    c = blob[ctl:ctl + 0x800]
    w = lambda o: struct.unpack_from("<I", c, o)[0]
    if (w(0xC), w(0x10)) != (0x120, 0xB) or w(0x708) not in (0x30, 0x100) or tag + w(0x704) > ctl:
        return None
    text = b""
    for magic in (b"!!NVvp5.0", b"!!NVfp5.0", b"!!NVgp5.0", b"!!NVcp5.0", b"!!NVtcp5.0", b"!!NVtep5.0"):
        i = blob.find(magic)
        if i >= 0:
            text = blob[i:].split(b"\0")[0]
            break
    return blob[tag:tag + w(0x704)], c, text


# --- .pak (version 11): the global shader cache ------------------------------------------------------------

def pak_file(pak_path, suffix):
    """(method name, block sizes, block size, data offset, unpacked size) of the pak entry ending in `suffix`."""
    with open(pak_path, "rb") as pak:
        pak.seek(0, 2)
        end = pak.tell()
        footer_size = 16 + 1 + 4 + 4 + 8 + 8 + 20 + 5 * 32
        pak.seek(end - footer_size)
        f = pak.read(footer_size)
        magic, version, index_off, index_size = struct.unpack_from("<IIqq", f, 17)
        if magic != 0x5A6F12E1 or version != 11:
            sys.exit(f"{pak_path}: not a version 11 pak (magic 0x{magic:x}, version {version})")
        methods = [f[61 + 32 * i:93 + 32 * i].split(b"\0")[0].decode() for i in range(5)]
        pak.seek(index_off)
        ix = pak.read(index_size)

        class R:
            def __init__(s, b): s.b, s.o = b, 0
            def u(s, fmt):
                v = struct.unpack_from("<" + fmt, s.b, s.o)
                s.o += struct.calcsize("<" + fmt)
                return v[0]
            def fstr(s):
                n = s.u("i")
                if n < 0:
                    raw = s.b[s.o:s.o - 2 * n]; s.o -= 2 * n
                    return raw.decode("utf-16-le").rstrip("\0")
                raw = s.b[s.o:s.o + n]; s.o += n
                return raw.decode("latin-1").rstrip("\0")
            def raw(s, n):
                v = s.b[s.o:s.o + n]; s.o += n
                return v

        r = R(ix)
        mount = r.fstr(); r.u("i"); r.u("Q")
        if r.u("I"):
            r.u("q"); r.u("q"); r.raw(20)
        if not r.u("I"):
            sys.exit(f"{pak_path}: no full directory index")
        full_off, full_size = r.u("q"), r.u("q"); r.raw(20)
        encoded = r.raw(r.u("i"))
        pak.seek(full_off)
        d = R(pak.read(full_size))
        where = None
        for _ in range(d.u("i")):
            dname = d.fstr()
            for _ in range(d.u("i")):
                fname = d.fstr()
                loc = d.u("i")
                if (mount + dname + fname).endswith(suffix):
                    where = loc
        if where is None:
            sys.exit(f"{pak_path}: no {suffix}")
        e = R(encoded); e.o = where
        v = e.u("I")
        block_size = e.u("I") if (v & 0x3F) == 0x3F else (v & 0x3F) << 11
        method = (v >> 23) & 0x3F
        nblocks = (v >> 6) & 0xFFFF
        off = e.u("I") if v & (1 << 31) else e.u("q")
        usize = e.u("I") if v & (1 << 30) else e.u("q")
        size = (e.u("I") if v & (1 << 29) else e.u("q")) if method else usize
        if (v >> 22) & 1:
            sys.exit(f"{pak_path}: {suffix} is encrypted")
        blocks = [] if not method else [size] if nblocks == 1 else [e.u("I") for _ in range(nblocks)]
        header = 8 + 8 + 8 + 4 + 20 + (4 + 16 * nblocks if method else 0) + 1 + 4
        return (methods[method - 1] if method else "none"), blocks, block_size, off + header, usize


def oodle_decompress(dll, blocks, block_size, usize):
    """Oodle blocks decompressed with the user's oo2core DLL (Windows only)."""
    import ctypes
    lib = ctypes.WinDLL(dll)
    f = lib.OodleLZ_Decompress
    f.restype = ctypes.c_ssize_t
    f.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_ssize_t] + [ctypes.c_int] * 3 + \
        [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]
    out = bytearray()
    for i, src in enumerate(blocks):
        want = min(block_size, usize - len(out))
        dst = ctypes.create_string_buffer(want)
        if f(src, len(src), dst, want, 1, 0, 0, None, 0, None, None, None, 0, 3) != want:
            sys.exit(f"Oodle: block {i} did not decompress")
        out += dst.raw
    return bytes(out)


def global_cache(pak, oodle, windows_python, work):
    """The unpacked GlobalShaderCache-*.bin of the pak."""
    method, sizes, block_size, at, usize = pak_file(pak, "GlobalShaderCache-SF_GLSL_SWITCH.bin")
    with open(pak, "rb") as f:
        f.seek(at)
        blocks = [f.read(n) for n in sizes] if sizes else [f.read(usize)]
    if method == "none":
        return blocks[0]
    if method == "Zlib":
        return b"".join(zlib.decompress(b) for b in blocks)
    if method != "Oodle":
        sys.exit(f"{pak}: the global shader cache is packed with {method}")
    if os.name == "nt":
        return oodle_decompress(oodle, blocks, block_size, usize)
    # WSL: the DLL loads only in Windows Python; hand it the blocks
    os.makedirs(work, exist_ok=True)
    raw = os.path.join(work, "gsc.blocks")
    with open(raw, "wb") as f:
        for b in blocks:
            f.write(struct.pack("<I", len(b)) + b)
    out = os.path.join(work, "gsc.bin")
    win = lambda p: subprocess.check_output(["wslpath", "-w", os.path.abspath(p)], text=True).strip()
    subprocess.check_call([windows_python, "-I", win(__file__), "oodle", win(oodle), win(raw),
                           str(block_size), str(usize), win(out)], cwd="/mnt/c")
    with open(out, "rb") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stages")
    st.add_argument("chunks"); st.add_argument("pak"); st.add_argument("out")
    st.add_argument("--oodle", default=os.path.join(os.path.dirname(__file__), "..", "retoc", "oo2core_9_win64.dll"))
    st.add_argument("--windows-python", default="python.exe")
    st.add_argument("--global-only", action="store_true", help="only the global shader cache (into an existing out dir)")
    od = sub.add_parser("oodle")  # internal: Windows Python's half under WSL
    od.add_argument("dll"); od.add_argument("blocks"); od.add_argument("block_size", type=int)
    od.add_argument("usize", type=int); od.add_argument("out")
    a = ap.parse_args()

    if a.cmd == "oodle":
        blocks, data = [], open(a.blocks, "rb").read()
        i = 0
        while i < len(data):
            n = struct.unpack_from("<I", data, i)[0]
            blocks.append(data[i + 4:i + 4 + n]); i += 4 + n
        open(a.out, "wb").write(oodle_decompress(a.dll, blocks, a.block_size, a.usize))
        return

    os.makedirs(a.out, exist_ok=True)
    seen, entries_n, bad = set(), 0, 0
    written = {f[:-5] for f in os.listdir(a.out) if f.endswith(".ctrl")}  # stages already there (a resumed run)
    def take(blob):
        nonlocal entries_n, bad
        entries_n += 1
        parts = split(blob)
        if not parts:
            bad += 1
            return
        code, ctrl, text = parts
        h = hashlib.sha1(code + ctrl).hexdigest()[:16]
        if h in written:
            return
        written.add(h)
        for ext, b in ((".code", code), (".ctrl", ctrl), (".asm", text)):
            with open(os.path.join(a.out, h + ext), "wb") as f:
                f.write(b)
    files = [] if a.global_only else [os.path.join(d, f) for d, _, fs in os.walk(a.chunks) for f in fs]
    for i, path in enumerate(files):
        with open(path, "rb") as f:
            data = f.read()
        for blob in entries(data, seen):
            take(blob)
        if (i + 1) % 5000 == 0:
            print(f"{i + 1} of {len(files)} chunks, {len(written)} stages", flush=True)
    work = os.path.join(os.path.abspath(a.out), "..", "shaderseed_work")
    for blob in entries(global_cache(a.pak, a.oodle, a.windows_python, work), seen):
        take(blob)
    print(f"{len(files)} chunks + the global shader cache: {entries_n} distinct entries, {len(written)} distinct "
          f"stages in {a.out}; {bad} entries not split")


if __name__ == "__main__":
    main()
