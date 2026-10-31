"""Groups coverage.cpp's unhandled-encoding TSV by mnemonic using Capstone.

Usage: label.py <unhandled.tsv> [top_n]

Encodings Capstone cannot decode are reported as "<invalid>"; these are
usually data embedded in .text rather than real instructions.
"""
import collections
import struct
import sys

import capstone


def main():
    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    by_mnemonic = collections.Counter()
    distinct = collections.Counter()
    sample = {}
    total = 0
    with open(path) as f:
        for line in f:
            enc_hex, count, offset = line.split("\t")
            enc, count = int(enc_hex, 16), int(count)
            insn = next(md.disasm(struct.pack("<I", enc), 0), None)
            key = insn.mnemonic if insn else "<invalid>"
            by_mnemonic[key] += count
            distinct[key] += 1
            total += count
            if key not in sample or count > sample[key][0]:
                text = f"{insn.mnemonic} {insn.op_str}" if insn else enc_hex
                sample[key] = (count, f"{text}  @0x{offset.strip()}")
    print(f"{'mnemonic':<14}{'count':>10}{'share':>8}{'encs':>7}  most common form")
    for key, count in by_mnemonic.most_common(top_n):
        print(f"{key:<14}{count:>10}{100 * count / total:>7.2f}%{distinct[key]:>7}  {sample[key][1]}")
    rest = sum(c for k, c in by_mnemonic.most_common()[top_n:])
    if rest:
        print(f"{'(other)':<14}{rest:>10}{100 * rest / total:>7.2f}%")


if __name__ == "__main__":
    main()
