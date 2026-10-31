"""Checks a captured hello-world frame: text in the top-left corner, black elsewhere.

Usage: check_frame.py <frame.png>

Decodes the PNG with the standard library only: 8-bit RGB or RGBA (the runtime
writes PNGs through SDL, which stores RGBA), any of the five row filters;
decode() returns RGB rows.
"""
import struct
import sys
import zlib


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def decode(path):
    """(width, height, rows): each row is width x 3 bytes of RGB."""
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path}: not a PNG")
    pos, idat, width, height, channels = 8, b"", 0, 0, 3
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
            if depth != 8 or color not in (2, 6) or body[12] != 0:
                raise SystemExit(f"{path}: expected 8-bit RGB or RGBA, not interlaced")
            channels = 3 if color == 2 else 4
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * channels
    rows, prev = [], bytearray(stride)
    for y in range(height):
        f = raw[y * (stride + 1)]
        cur = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = cur[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            if f == 1: cur[i] = (cur[i] + a) & 0xFF
            elif f == 2: cur[i] = (cur[i] + b) & 0xFF
            elif f == 3: cur[i] = (cur[i] + ((a + b) >> 1)) & 0xFF
            elif f == 4: cur[i] = (cur[i] + _paeth(a, b, c)) & 0xFF
            elif f != 0: raise SystemExit(f"{path}: bad PNG filter {f}")
        prev = cur
        rows.append(bytes(cur) if channels == 3 else bytes(v for x in range(width) for v in cur[4 * x:4 * x + 3]))
    return width, height, rows


def main():
    width, height, rows = decode(sys.argv[1])
    lit = [(x, y) for y, row in enumerate(rows) for x in range(width)
           if row[3 * x] | row[3 * x + 1] | row[3 * x + 2]]
    if (width, height) != (1280, 720):
        raise SystemExit(f"FAIL: frame is {width}x{height}, expected 1280x720")
    if not lit:
        raise SystemExit("FAIL: frame is entirely black")
    max_x = max(x for x, _ in lit)
    max_y = max(y for _, y in lit)
    # "Hello World!" is 12 cells of the 8x16 console font on row 0.
    if max_x >= 12 * 8 * 2 or max_y >= 16 or not 150 <= len(lit) <= 2000:
        raise SystemExit(f"FAIL: unexpected content: {len(lit)} lit pixels, extent {max_x + 1}x{max_y + 1}")
    print(f"frame OK: {len(lit)} lit pixels within {max_x + 1}x{max_y + 1} at the top left, rest black")


if __name__ == "__main__":
    main()
