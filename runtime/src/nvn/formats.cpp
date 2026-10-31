// NVN texture formats (NVN_PLAN 6.1): bytes per block and block size per NVN
// format id (the ids are the same in SDK 3.5.0 to 12.3.2).
//
// The sizes follow LASRecomp's formats.c (github.com/peiikai/LASRecomp,
// 6244eb9, MIT licence), except 0x95-0x9c (from the SDK's table, below). The title route's 27 formats
// are checked against the driver that way; the rest are unchecked.
#include "nvn.h"

#include <array>

namespace rt::nvn {

namespace {

constexpr FormatInfo C(u8 bpb) { return {bpb, 1, 1, false}; }
constexpr FormatInfo B(u8 bpb, u8 w, u8 h) { return {bpb, w, h, false}; }
constexpr FormatInfo Z(u8 bpb) { return {bpb, 1, 1, true}; }

constexpr std::array<FormatInfo, 0x9d> MakeTable() {
    std::array<FormatInfo, 0x9d> t{};
    auto put = [&](u32 first, u32 last, FormatInfo f) {
        for (u32 i = first; i <= last; ++i) t[i] = f;
    };
    put(0x01, 0x04, C(1));  // R8
    put(0x05, 0x09, C(2));  // R16
    put(0x0a, 0x0c, C(4));  // R32
    put(0x0d, 0x10, C(2));  // RG8
    put(0x11, 0x15, C(4));  // RG16
    put(0x16, 0x18, C(8));  // RG32
    put(0x19, 0x1c, C(3));  // RGB8
    put(0x1d, 0x21, C(6));  // RGB16
    put(0x22, 0x24, C(12)); // RGB32
    put(0x25, 0x28, C(4));  // RGBA8
    put(0x29, 0x2d, C(8));  // RGBA16
    put(0x2e, 0x30, C(16)); // RGBA32
    t[0x31] = Z(1);         // stencil 8
    t[0x32] = Z(2);         // depth 16
    put(0x33, 0x35, Z(4));  // depth 24, depth 32F, depth 24 stencil 8
    t[0x36] = Z(8);         // depth 32F stencil 8
    put(0x37, 0x38, C(4));  // RGBX8 / RGBA8 sRGB
    put(0x39, 0x3c, C(2));  // RGBA4, RGB5, RGB5A1, RGB565
    put(0x3d, 0x40, C(4));  // RGB10A2(UI), R11G11B10F, RGB9E5
    for (u32 i = 0x41; i <= 0x50; ++i) { // BC1-BC7
        const bool eight = i == 0x41 || i == 0x42 || i == 0x45 || i == 0x46 || i == 0x49 || i == 0x4a;
        t[i] = B(eight ? 8 : 16, 4, 4);
    }
    put(0x51, 0x52, C(1));  // integer-to-float vertex formats
    put(0x53, 0x54, C(2));
    put(0x55, 0x56, C(4));
    put(0x57, 0x58, C(2));
    put(0x59, 0x5a, C(4));
    put(0x5b, 0x5c, C(8));
    put(0x5d, 0x5e, C(3));
    put(0x5f, 0x60, C(6));
    put(0x61, 0x62, C(12));
    put(0x63, 0x64, C(4));
    put(0x65, 0x66, C(8));
    put(0x67, 0x68, C(16));
    put(0x69, 0x70, C(4));  // ... and RGBX8
    put(0x71, 0x75, C(8));  // RGBX16
    put(0x76, 0x78, C(16)); // RGBX32
    constexpr u8 astc[14][2] = {{4, 4},  {5, 4},  {5, 5},  {6, 5},   {6, 6},   {8, 5},   {8, 6},
                                {8, 8},  {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12}};
    for (u32 k = 0; k < 14; ++k) { // ASTC, linear then sRGB
        t[0x79 + k] = B(16, astc[k][0], astc[k][1]);
        t[0x87 + k] = B(16, astc[k][0], astc[k][1]);
    }
    // 0x95-0x9c from BRO's own format table (.data 0xd777c8, sdk-rea dump; its sizes agree with this table
    // elsewhere): four 16-bit formats (0x98: 5-5-5-1), then BGRX8, BGRA8, BGRX8 sRGB, BGRA8 sRGB — one id
    // later than LASRecomp's table (which has 0x98 = BGRX8). The window code agrees: it groups 0x9a with 0x9c.
    put(0x95, 0x98, C(2));
    put(0x99, 0x9c, C(4));
    return t;
}

constexpr auto kFormats = MakeTable();

} // namespace

const FormatInfo* FindFormat(u32 id) {
    return id < kFormats.size() && kFormats[id].bytes ? &kFormats[id] : nullptr;
}

} // namespace rt::nvn
