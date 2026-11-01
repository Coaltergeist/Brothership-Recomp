// Stand-in for MSVC's <intrin.h> as ooz uses it (seed/ooz.cpp): SSE2 and the bit helpers. MSVC's unsigned long is
// 32 bits; ooz passes its own unsigned long variables, so the wider type here is harmless.
#pragma once

#include <immintrin.h>
#include <stdint.h>

static inline unsigned char _BitScanReverse(unsigned long* i, unsigned long m) {
    if (!m) return 0;
    *i = 31 - __builtin_clz(static_cast<unsigned>(m));
    return 1;
}
static inline unsigned char _BitScanForward(unsigned long* i, unsigned long m) {
    if (!m) return 0;
    *i = __builtin_ctz(static_cast<unsigned>(m));
    return 1;
}
static inline uint16_t _byteswap_ushort(uint16_t x) { return __builtin_bswap16(x); }
static inline uint32_t _byteswap_ulong(uint32_t x) { return __builtin_bswap32(x); }
static inline uint64_t _byteswap_uint64(uint64_t x) { return __builtin_bswap64(x); }
#ifndef _rotl // (GCC's <ia32intrin.h> has it as a macro)
static inline uint32_t _rotl(uint32_t x, int n) {
    n &= 31;
    return n ? (x << n) | (x >> (32 - n)) : x;
}
#endif
