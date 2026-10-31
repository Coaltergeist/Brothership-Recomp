# Which NVN function-table slots a game's main reads, and where (the calls the game can make).
# Slot names come from the game's own loader function, which asks for each name and stores the answer
# (adrp/add x1 = name string; blr; adrp + str x0 -> slot); reads are adrp + ldr (64-bit, unsigned offset)
# pairs on the same register anywhere in .text outside the loader (the last adrp per register: a load
# whose page comes another way is missed). Lists the slots read whose names are in <names file> (e.g. the
# layer's "nvn: not implemented:" log line, one name per line or space-separated).
# Brothership 1.0.1: loader at ELF 0x3070730 (Ghidra 0x3170730); 525 slots, 219 read (2026-10-23).
#
# Usage: python3 -I nvn_slot_refs.py <main.elf from tools/nso2elf> <loader ELF address> <names file>
import struct, sys
from collections import defaultdict

elf = open(sys.argv[1], 'rb').read()
loader = int(sys.argv[2], 16)
missing = set(open(sys.argv[3]).read().split())
phoff = struct.unpack_from('<Q', elf, 0x20)[0]; phn = struct.unpack_from('<H', elf, 0x38)[0]
segs = []
for i in range(phn):
    t, f, off, va, pa, fs, ms, al = struct.unpack_from('<IIQQQQQQ', elf, phoff + i * 56)
    if t == 1: segs.append((va, off, fs, f))
def at(va, n):
    for v, o, fs, f in segs:
        if v <= va < v + fs: return elf[o + va - v:o + va - v + n]
    raise KeyError(hex(va))
def cstr(va):
    b = at(va, 128); return b[:b.index(b'\0')].decode()
text = next(s for s in segs if s[3] & 1)
def adrp(pc, w): # page or None
    if (w & 0x9F000000) != 0x90000000: return None
    imm = (((w >> 5) & 0x7FFFF) << 2) | ((w >> 29) & 3)
    if imm & (1 << 20): imm -= 1 << 21
    return (pc & ~0xFFF) + (imm << 12)

# the loader: name in x1, result stored to a slot
slots = {}
reg = {}
pc = loader
while True:
    w = struct.unpack('<I', at(pc, 4))[0]
    if w == 0xD65F03C0: break # ret
    p = adrp(pc, w)
    if p is not None: reg[w & 31] = p
    elif (w & 0xFFC00000) == 0x91000000: # add xd, xn, #imm
        rd, rn, imm = w & 31, (w >> 5) & 31, (w >> 10) & 0xFFF
        if rn in reg: reg[rd] = reg[rn] + imm
    elif (w & 0xFFC00000) == 0xF9000000: # str xt, [xn, #imm*8]
        rt, rn, imm = w & 31, (w >> 5) & 31, ((w >> 10) & 0xFFF) * 8
        if rt == 0 and rn in reg and 1 in reg: slots[reg[rn] + imm] = cstr(reg[1])
    pc += 4
    if pc - loader > 0x10000: sys.exit('no ret in the loader')
loader_end = pc
print(f'{len(slots)} slots named by the loader (0x{min(slots):x}-0x{max(slots):x})')

refs = defaultdict(list)
va, off, fs, _ = text
words = struct.unpack_from(f'<{fs // 4}I', elf, off)
last = {}
for i, w in enumerate(words):
    pc = va + 4 * i
    if (w & 0x9F000000) == 0x90000000:
        last[w & 31] = adrp(pc, w); continue
    if (w & 0xFFC00000) == 0xF9400000: # ldr xt, [xn, #imm*8]
        rn = (w >> 5) & 31
        if rn in last:
            a = last[rn] + ((w >> 10) & 0xFFF) * 8
            if a in slots and not (loader <= pc <= loader_end): refs[slots[a]].append(pc)
print(f'{len(refs)} slots read outside the loader')
hit = sorted(n for n in refs if n in missing)
print(f'of them not implemented in the layer: {len(hit)}')
for n in hit: print(f'  {n}: ' + ' '.join(f'0x{p:x}' for p in refs[n][:8]) + (' ...' if len(refs[n]) > 8 else ''))
never = sorted(n for n in slots.values() if n not in refs)
print(f'slots never read: {len(never)}')
