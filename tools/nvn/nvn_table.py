# NVN entry-point name -> sdk function, read statically the way sdk's default
# nvnDeviceGetProcAddress (sdk+0x46cb00, Brothership's SDK) does it: a sorted
# table of names without the "nvn" prefix (u32 offsets at sdk+0xad055c into a
# string pool at sdk+0xaccd57, 0x20c+1 entries), binary-searched with strcmp;
# the function is default_dispatch_table[index], the table found through the
# GOT slot sdk+0xc8e758. The addresses are specific to this sdk build: check
# them against a disassembly of sdk+0x46cb00 before using this on another title.
#
# Usage: python -I nvn_table.py <nxrecomp out>/image.bin <out.tsv>
# (a linked image, sdk at +0x7d31000, base 0x8000000; see link.txt)
import sys, struct
img_path, out_path = sys.argv[1:3]
BASE, SDK = 0x8000000, 0x7d31000
img = open(img_path, 'rb').read()
u32 = lambda o: struct.unpack_from('<I', img, SDK + o)[0]
u64 = lambda o: struct.unpack_from('<Q', img, SDK + o)[0]
def cstr(o):
    e = img.index(b'\0', SDK + o); return img[SDK + o:e].decode()
names = ['nvn' + cstr(0xaccd57 + u32(0xad055c + 4 * i)) for i in range(0x20c + 1)]
assert names == sorted(names, key=lambda s: s.encode()), 'not sorted: wrong table?'
table = u64(0xc8e758) - BASE - SDK
syms = {}
for l in open(img_path.replace('image.bin', 'symbols.txt')):
    a, n = l.split(' ', 1); syms[int(a, 16)] = n.strip()
rows = []
for i, n in enumerate(names):
    f = u64(table + 8 * i)
    rows.append((n, f - BASE - SDK if f else None))
with open(out_path, 'w') as fo:
    fo.write('# name\tsdk offset (from sdk default dispatch table, read statically)\n')
    for n, f in rows: fo.write('%s\t%s\n' % (n, '0x%x' % f if f is not None else '-'))
fs = [f for _, f in rows if f is not None]
print('names %d, table at sdk+0x%x, non-null %d, distinct %d, inside sdk .text %d' % (
    len(names), table, len(fs), len(set(fs)), sum(0 <= f < 0x58f710 for f in fs)))
