"""Retargets an ELF64 version requirement to a version the host already provides.

Usage: retarget_verneed.py <old_version> <new_version> <elf>...

Rewrites every .gnu.version_r entry naming <old_version> (e.g. GLIBC_2.36) to
name <new_version> (e.g. GLIBC_2.34), whose string must already be in the file's
dynamic string table. Only safe once no symbol is bound to the old version
(see patchelf --clear-symbol-version); otherwise the loader would bind that
symbol against the wrong version.
"""
import struct
import sys


def elf_hash(name):
    h = 0
    for c in name.encode():
        h = ((h << 4) + c) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def retarget(path, version, replacement):
    data = bytearray(open(path, "rb").read())
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return False
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    loads, dyn = [], None
    for i in range(phnum):
        p_type, _, p_offset, p_vaddr, _, p_filesz, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + i * phentsize)
        if p_type == 1:
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == 2:
            dyn = (p_offset, p_filesz)
    if dyn is None:
        return False

    def off(va):
        return next(o + va - v for v, o, s in loads if v <= va < v + s)

    tags = {}
    for i in range(0, dyn[1], 16):
        t, v = struct.unpack_from("<qQ", data, dyn[0] + i)
        if t == 0:
            break
        tags[t] = v
    if 0x6FFFFFFE not in tags:  # DT_VERNEED
        return False
    strtab, verneed, count = off(tags[5]), off(tags[0x6FFFFFFE]), tags[0x6FFFFFFF]
    target = data.find(replacement.encode() + b"\0", strtab)
    if target < 0:
        raise SystemExit(f"{path}: no {replacement} string to retarget to")

    changed = False
    for _ in range(count):
        _, cnt, _, aux, nxt = struct.unpack_from("<HHIII", data, verneed)
        a = verneed + aux
        for _ in range(cnt):
            _, _, _, name, anext = struct.unpack_from("<IHHII", data, a)
            if data[strtab + name:data.index(0, strtab + name)].decode() == version:
                struct.pack_into("<I", data, a, elf_hash(replacement))  # vna_hash
                struct.pack_into("<I", data, a + 8, target - strtab)    # vna_name
                changed = True
            a += anext
        if not nxt:
            break
        verneed += nxt
    if changed:
        open(path, "wb").write(data)
    return changed


def main():
    old, new, *paths = sys.argv[1:]
    for p in paths:
        print(("retargeted " if retarget(p, old, new) else "unchanged  ") + p)


if __name__ == "__main__":
    main()
