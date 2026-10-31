"""Converts an uncompressed NSO (hactool --uncompressed) into an AArch64 ELF
shared object, so tools that only read ELF (REA, Ghidra's ELF loader, objdump,
IDA) can analyse Switch modules.

Defined .dynsym entries get their section index rewritten to this file's
sections (the NSO keeps the original ELF's indices, which no longer apply);
nothing else in the image changes.

The ELF is ET_DYN at virtual address 0: every address is module-relative, as in
our notes (sdk+X), and a loader that rebases it (Ghidra puts PIEs at 0x100000)
applies the module's own RELA relocations. Program headers: one PT_LOAD per
NSO segment (.bss in the data segment's memsz), PT_DYNAMIC and PT_GNU_EH_FRAME
from MOD0. Section headers name .text, .rodata, .data, .bss, .dynamic,
.dynsym, .dynstr, .eh_frame_hdr and .eh_frame, so analysers that look for
those sections (Ghidra's exception-handler analysis) find them.

PLT stubs are always named <symbol>@plt in the .symtab.

Usage: python -I nso2elf.py <module.bin> <out.elf> [names.tsv]

names.tsv adds a .symtab of extra function names ("name<TAB>0xoffset" lines,
'#' comments, '-' offsets skipped), e.g. tools/nvn/nvn_table.py's output.
"""
import struct
import sys

PT_LOAD, PT_DYNAMIC, PT_GNU_EH_FRAME = 1, 2, 0x6474E550
PF_X, PF_W, PF_R = 1, 2, 4
SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, SHT_DYNAMIC, SHT_NOBITS, SHT_DYNSYM = 1, 2, 3, 6, 8, 11
SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR = 1, 2, 4
DT_PLTRELSZ, DT_STRTAB, DT_SYMTAB, DT_STRSZ, DT_JMPREL = 2, 5, 6, 10, 23


def load_nso(data):
    if data[:4] != b"NSO0":
        raise SystemExit("not an NSO")
    if struct.unpack_from("<I", data, 0xC)[0] & 7:
        raise SystemExit("segments are compressed: extract with hactool --uncompressed")
    segs = []
    for i in range(3):
        file_off, mem_off, size = struct.unpack_from("<III", data, 0x10 + 0x10 * i)
        segs.append((file_off, mem_off, size))
    bss = struct.unpack_from("<I", data, 0x3C)[0]
    end = max(m + s for _, m, s in segs)
    mem = bytearray(end)
    for f, m, s in segs:
        mem[m:m + s] = data[f:f + s]
    return mem, [(m, s) for _, m, s in segs], bss


def eh_frame_size(mem, start):
    """Length of the .eh_frame records starting at start (up to the zero terminator)."""
    at = start
    while at + 4 <= len(mem):
        length = struct.unpack_from("<I", mem, at)[0]
        if length == 0:
            return at + 4 - start
        if length == 0xFFFFFFFF:
            length = struct.unpack_from("<Q", mem, at + 4)[0] + 8
        at += 4 + length
    return at - start


def main():
    src, dst = sys.argv[1:3]
    extra = []
    if len(sys.argv) > 3:
        for line in open(sys.argv[3]):
            f = line.rstrip("\n").split("\t")
            if line.startswith("#") or len(f) < 2 or f[1] == "-":
                continue
            extra.append((f[0], int(f[1], 16)))
    mem, segs, bss = load_nso(open(src, "rb").read())
    (text, text_size), (ro, ro_size), (data_off, data_size) = segs

    mod0 = struct.unpack_from("<I", mem, text + 4)[0]
    if mem[mod0:mod0 + 4] != b"MOD0":
        raise SystemExit("no MOD0 header")
    rel = lambda field: mod0 + struct.unpack_from("<i", mem, mod0 + field)[0]
    dynamic, ehdr_begin, ehdr_end = rel(4), rel(16), rel(20)
    dyn = {}
    at = dynamic
    while True:
        tag, val = struct.unpack_from("<qQ", mem, at)
        at += 16
        if tag == 0:
            break
        dyn.setdefault(tag, val)
    dynamic_size = at - dynamic
    eh_frame = ehdr_begin + 4 + struct.unpack_from("<i", mem, ehdr_begin + 4)[0]  # pcrel sdata4
    eh_frame_len = eh_frame_size(mem, eh_frame)

    # File layout: ELF header and program headers in the first page, then the
    # module image at file offset 0x1000 + vaddr, then section headers.
    phdrs = [
        (PT_LOAD, PF_R | PF_X, text, text_size, text_size, 0x1000),
        (PT_LOAD, PF_R, ro, ro_size, ro_size, 0x1000),
        (PT_LOAD, PF_R | PF_W, data_off, data_size, data_size + bss, 0x1000),
        (PT_DYNAMIC, PF_R | PF_W, dynamic, dynamic_size, dynamic_size, 8),
        (PT_GNU_EH_FRAME, PF_R, ehdr_begin, ehdr_end - ehdr_begin, ehdr_end - ehdr_begin, 4),
    ]
    base = 0x1000
    # Section indices in .dynsym refer to the original ELF's section table,
    # which the NSO does not keep: point each defined symbol at the section of
    # this file that contains it (1 .text, 2 .rodata, 3 .data, 4 .bss).
    # Ghidra places symbols by section, so stale indices misplace all names.
    for at in range(dyn[DT_SYMTAB], dyn[DT_STRTAB] - 23, 24):
        shndx, value = struct.unpack_from("<H", mem, at + 6)[0], struct.unpack_from("<Q", mem, at + 8)[0]
        if shndx == 0 or shndx >= 0xFF00:
            continue  # undefined, ABS, COMMON
        if text <= value < text + text_size:
            new = 1
        elif ro <= value < ro + ro_size:
            new = 2
        elif data_off <= value < data_off + data_size:
            new = 3
        elif data_off + data_size <= value <= data_off + data_size + bss:
            new = 4
        else:
            continue
        struct.pack_into("<H", mem, at + 6, new)
    image = bytes(mem[:data_off + data_size])

    names = b"\0"
    def name(s):
        nonlocal names
        off = len(names)
        names += s.encode() + b"\0"
        return off
    sym_count = (dyn[DT_STRTAB] - dyn[DT_SYMTAB]) // 24  # dynsym conventionally ends at dynstr
    # (name, type, flags, addr, size, link, info, align, entsize)
    sections = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (name(".text"), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text, text_size, 0, 0, 16, 0),
        (name(".rodata"), SHT_PROGBITS, SHF_ALLOC, ro, ro_size, 0, 0, 16, 0),
        (name(".data"), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, data_off, data_size, 0, 0, 16, 0),
        (name(".bss"), SHT_NOBITS, SHF_ALLOC | SHF_WRITE, data_off + data_size, bss, 0, 0, 16, 0),
        (name(".dynstr"), SHT_STRTAB, SHF_ALLOC, dyn[DT_STRTAB], dyn[DT_STRSZ], 0, 0, 1, 0),
    ]
    dynstr_index = len(sections) - 1
    sections += [
        (name(".dynsym"), SHT_DYNSYM, SHF_ALLOC, dyn[DT_SYMTAB], sym_count * 24, dynstr_index, 1, 8, 24),
        (name(".dynamic"), SHT_DYNAMIC, SHF_ALLOC | SHF_WRITE, dynamic, dynamic_size, dynstr_index, 0, 8, 16),
        (name(".eh_frame_hdr"), SHT_PROGBITS, SHF_ALLOC, ehdr_begin, ehdr_end - ehdr_begin, 0, 0, 4, 0),
        (name(".eh_frame"), SHT_PROGBITS, SHF_ALLOC, eh_frame, eh_frame_len, 0, 0, 8, 0),
    ]
    # PLT stubs ("adrp x16, P; ldr x17, [x16, #o]; add x16, x16, #o; br x17")
    # are named <symbol>@plt after the JUMP_SLOT relocation of their GOT slot.
    # (Section headers for .plt/.got.plt would overlap .text/.data, which
    # Ghidra's ELF loader does not accept.)
    if DT_JMPREL in dyn:
        slot_name = {}
        dynstr = dyn[DT_STRTAB]
        for r in range(dyn[DT_JMPREL], dyn[DT_JMPREL] + dyn[DT_PLTRELSZ], 24):
            off, info = struct.unpack_from("<QQ", mem, r)
            sym = dyn[DT_SYMTAB] + 24 * (info >> 32)
            n = struct.unpack_from("<I", mem, sym)[0]
            slot_name[off] = mem[dynstr + n:mem.index(0, dynstr + n)].decode()
        for pc in range(text, text + text_size - 12, 4):
            adrp, ldr, add, br = struct.unpack_from("<IIII", mem, pc)
            if (adrp & 0x9F00001F) != 0x90000010 or (ldr & 0xFFC003FF) != 0xF9400211 or br != 0xD61F0220:
                continue
            imm = (((adrp >> 5) & 0x7FFFF) << 2) | ((adrp >> 29) & 3)
            if imm & (1 << 20):
                imm -= 1 << 21
            slot = (pc & ~0xFFF) + (imm << 12) + ((ldr >> 10) & 0xFFF) * 8
            if slot in slot_name and slot_name[slot]:
                extra.append((slot_name[slot] + "@plt", pc))

    # Optional .symtab/.strtab (not loaded) after the image.
    tail = bytearray()
    files = {}  # section name offset -> file offset of a non-alloc section
    if extra:
        strtab = bytearray(b"\0")
        symtab = bytearray(24)
        for n, off in extra:
            symtab += struct.pack("<IBBHQQ", len(strtab), 0x12, 0, 1, off, 0)  # GLOBAL FUNC in .text
            strtab += n.encode() + b"\0"
        strtab_name, symtab_name = name(".strtab"), name(".symtab")
        files[strtab_name] = base + len(image)
        tail += strtab
        tail += bytes(-len(tail) % 8)
        files[symtab_name] = base + len(image) + len(tail)
        tail += symtab
        sections.append((strtab_name, SHT_STRTAB, 0, 0, len(strtab), 0, 0, 1, 0))
        sections.append((symtab_name, SHT_SYMTAB, 0, 0, len(symtab), len(sections) - 1, 1, 8, 24))
    shstr_name = name(".shstrtab")
    shstr_off = base + len(image) + len(tail)
    sh_off = (shstr_off + len(names) + 7) & ~7
    sections.append((shstr_name, SHT_STRTAB, 0, 0, len(names), 0, 0, 1, 0))

    out = bytearray()
    out += b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    out += struct.pack("<HHIQQQIHHHHHH", 3, 183, 1, text, 64, sh_off, 0, 64, 56,
                       len(phdrs), 64, len(sections), len(sections) - 1)
    for p_type, flags, vaddr, filesz, memsz, align in phdrs:
        out += struct.pack("<IIQQQQQQ", p_type, flags, base + vaddr, vaddr, vaddr, filesz, memsz, align)
    out += bytes(base - len(out))
    out += image
    out += tail
    out += names
    out += bytes(sh_off - len(out))
    for n, t, flags, addr, size, link, info, align, entsize in sections:
        if t == SHT_STRTAB and n == shstr_name:
            off = shstr_off
        elif n in files:
            off = files[n]
        elif t == SHT_NOBITS or t == 0:
            off = base + addr if t else 0
        else:
            off = base + addr
        out += struct.pack("<IIQQQQIIQQ", n, t, flags, addr, off, size, link, info, align, entsize)
    open(dst, "wb").write(out)
    print(f"{dst}: {len(out):#x} bytes; .text {text:#x}+{text_size:#x}, .eh_frame {eh_frame:#x}+{eh_frame_len:#x}, "
          f"{sym_count} dynamic symbols, {len(extra)} extra names")


if __name__ == "__main__":
    main()
