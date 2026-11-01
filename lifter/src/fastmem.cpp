#include "fastmem.h"

namespace nxrecomp::fastmem {

const std::string& Header() {
    static const std::string h = R"C(#pragma once
/* nx_mem.h (lifter/src/fastmem.cpp): guest loads and stores, the common case inline.

   The fast path is the one recomp_host_ptr_n takes in recomp_runtime.c, with the same tests: a host
   page table, an address inside the address space whose bytes stay in one page, a page tagged plain
   memory with a backing pointer, and no access counters (they count every access). It assumes the
   runtime's table layout, 4 KB pages and one pointer-sized entry each (runtime/src/rt.cpp,
   Memory::EnableFastmem). Anything else calls drippu's helper by its own name (the parenthesised
   spelling skips the macro), which decides exactly as before: the emulator callback for unmapped, debug
   or GPU-tracked pages and for page-crossing accesses. Only the test and the access are inline, so the
   code stays small (drippu measured whole helpers inline at every site as slower: instruction cache). */

static inline __attribute__((always_inline)) unsigned char* nx_mem_ptr(GuestContext* c, uint64_t va, uint64_t bytes) {
    const RecompHostMem* hm = c->host_mem;
    uintptr_t raw, p;
    va &= 0xffffffffffffULL; /* AArch64 ignores the top 16 bits */
    if (__builtin_expect(!hm || !hm->page_entries || hm->counters, 0)) return 0;
    if (__builtin_expect(va >= hm->address_space_max || bytes > hm->address_space_max - va ||
                         (va & 0xFFF) > 0x1000 - bytes, 0)) return 0;
    raw = ((const uintptr_t*)hm->page_entries)[va >> 12];
    if (__builtin_expect((raw & (uintptr_t)hm->page_type_mask) != (uintptr_t)hm->page_type_memory, 0)) return 0;
    p = raw & (uintptr_t)hm->pointer_mask;
    return __builtin_expect(p != 0, 1) ? (unsigned char*)(p + (uintptr_t)va) : 0;
}

#define NX_MEM_LOAD(bits)                                                                  \
    static inline __attribute__((always_inline)) uint64_t nx_load##bits(GuestContext* c, uint64_t a) { \
        unsigned char* p = nx_mem_ptr(c, a, bits / 8);                                     \
        if (__builtin_expect(p != 0, 1)) {                                                 \
            uint##bits##_t v;                                                              \
            __builtin_memcpy(&v, p, bits / 8);                                             \
            return (uint64_t)v;                                                            \
        }                                                                                  \
        return (recomp_load##bits)(c, a);                                                  \
    }
#define NX_MEM_STORE(bits)                                                                 \
    static inline __attribute__((always_inline)) void nx_store##bits(GuestContext* c, uint64_t a, uint64_t v) { \
        unsigned char* p = nx_mem_ptr(c, a, bits / 8);                                     \
        if (__builtin_expect(p != 0, 1)) {                                                 \
            uint##bits##_t t = (uint##bits##_t)v;                                          \
            __builtin_memcpy(p, &t, bits / 8);                                             \
            return;                                                                        \
        }                                                                                  \
        (recomp_store##bits)(c, a, v);                                                     \
    }
NX_MEM_LOAD(8) NX_MEM_LOAD(16) NX_MEM_LOAD(32) NX_MEM_LOAD(64)
NX_MEM_STORE(8) NX_MEM_STORE(16) NX_MEM_STORE(32) NX_MEM_STORE(64)

/* Pairs: both words in one page (as recomp_pair_same_page), else drippu's pair helper. */
static inline __attribute__((always_inline)) void nx_ldp64(GuestContext* c, uint64_t a, uint64_t* lo, uint64_t* hi) {
    unsigned char* p = nx_mem_ptr(c, a, 16);
    if (__builtin_expect(p != 0, 1)) { __builtin_memcpy(lo, p, 8); __builtin_memcpy(hi, p + 8, 8); return; }
    (recomp_ldp64)(c, a, lo, hi);
}
static inline __attribute__((always_inline)) void nx_stp64(GuestContext* c, uint64_t a, uint64_t v0, uint64_t v1) {
    unsigned char* p = nx_mem_ptr(c, a, 16);
    if (__builtin_expect(p != 0, 1)) { __builtin_memcpy(p, &v0, 8); __builtin_memcpy(p + 8, &v1, 8); return; }
    (recomp_stp64)(c, a, v0, v1);
}
static inline __attribute__((always_inline)) void nx_ldp32(GuestContext* c, uint64_t a, uint64_t* lo, uint64_t* hi) {
    unsigned char* p = nx_mem_ptr(c, a, 8);
    if (__builtin_expect(p != 0, 1)) {
        uint32_t x, y;
        __builtin_memcpy(&x, p, 4); __builtin_memcpy(&y, p + 4, 4);
        *lo = (uint64_t)x; *hi = (uint64_t)y;
        return;
    }
    (recomp_ldp32)(c, a, lo, hi);
}
static inline __attribute__((always_inline)) void nx_stp32(GuestContext* c, uint64_t a, uint64_t v0, uint64_t v1) {
    unsigned char* p = nx_mem_ptr(c, a, 8);
    if (__builtin_expect(p != 0, 1)) {
        uint32_t x = (uint32_t)v0, y = (uint32_t)v1;
        __builtin_memcpy(p, &x, 4); __builtin_memcpy(p + 4, &y, 4);
        return;
    }
    (recomp_stp32)(c, a, v0, v1);
}

#define recomp_load8(c, a) nx_load8(c, a)
#define recomp_load16(c, a) nx_load16(c, a)
#define recomp_load32(c, a) nx_load32(c, a)
#define recomp_load64(c, a) nx_load64(c, a)
#define recomp_store8(c, a, v) nx_store8(c, a, v)
#define recomp_store16(c, a, v) nx_store16(c, a, v)
#define recomp_store32(c, a, v) nx_store32(c, a, v)
#define recomp_store64(c, a, v) nx_store64(c, a, v)
#define recomp_ldp32(c, a, lo, hi) nx_ldp32(c, a, lo, hi)
#define recomp_ldp64(c, a, lo, hi) nx_ldp64(c, a, lo, hi)
#define recomp_stp32(c, a, v0, v1) nx_stp32(c, a, v0, v1)
#define recomp_stp64(c, a, v0, v1) nx_stp64(c, a, v0, v1)
)C";
    return h;
}

const std::string& DirectHeader() {
    static const std::string h = R"C(#pragma once
/* nx_mem.h (lifter/src/fastmem.cpp, nxrecomp --direct-mem): guest loads and stores as plain host accesses.

   The runtime reserves the guest's 39-bit address space as one host range at NX_MEM_BASE (runtime/src/rt.h
   kHostWindowBase) and keeps every page the guest has not mapped inaccessible, so an access needs no test:
   NX_MEM_BASE plus the guest address. The mask is the guest's address space; it also drops the top-byte tag
   AArch64 ignores (TBI). A wild address wraps into the window (guest memory, never host data) and faults
   there unless mapped. GPU-tracked pages (the driver seam) are not supported: the runtime refuses that mode
   with this lift. Exclusives (recomp_ldxr ...) keep drippu's helpers. */

#define NX_MEM_BASE 0x400000000000ULL
#define NX_MEM_MASK 0x7FFFFFFFFFULL
typedef uint16_t nx_u16 __attribute__((aligned(1), may_alias));
typedef uint32_t nx_u32 __attribute__((aligned(1), may_alias));
typedef uint64_t nx_u64 __attribute__((aligned(1), may_alias));
#define NX_MEM_AT(a) ((uintptr_t)(NX_MEM_BASE + ((uint64_t)(a) & NX_MEM_MASK)))

#define recomp_load8(c, a) ((uint64_t)*(const uint8_t*)NX_MEM_AT(a))
#define recomp_load16(c, a) ((uint64_t)*(const nx_u16*)NX_MEM_AT(a))
#define recomp_load32(c, a) ((uint64_t)*(const nx_u32*)NX_MEM_AT(a))
#define recomp_load64(c, a) ((uint64_t)*(const nx_u64*)NX_MEM_AT(a))
#define recomp_store8(c, a, v) ((void)(*(uint8_t*)NX_MEM_AT(a) = (uint8_t)(v)))
#define recomp_store16(c, a, v) ((void)(*(nx_u16*)NX_MEM_AT(a) = (uint16_t)(v)))
#define recomp_store32(c, a, v) ((void)(*(nx_u32*)NX_MEM_AT(a) = (uint32_t)(v)))
#define recomp_store64(c, a, v) ((void)(*(nx_u64*)NX_MEM_AT(a) = (uint64_t)(v)))

/* Pairs: the two words are adjacent in the window (a pair crossing into an unmapped page faults there). */
static inline __attribute__((always_inline)) void nx_ldp64(uint64_t a, uint64_t* lo, uint64_t* hi) {
    const uintptr_t p = NX_MEM_AT(a);
    const uint64_t x = *(const nx_u64*)p, y = *(const nx_u64*)(p + 8);
    *lo = x; *hi = y;
}
static inline __attribute__((always_inline)) void nx_stp64(uint64_t a, uint64_t v0, uint64_t v1) {
    const uintptr_t p = NX_MEM_AT(a);
    *(nx_u64*)p = v0; *(nx_u64*)(p + 8) = v1;
}
static inline __attribute__((always_inline)) void nx_ldp32(uint64_t a, uint64_t* lo, uint64_t* hi) {
    const uintptr_t p = NX_MEM_AT(a);
    const uint32_t x = *(const nx_u32*)p, y = *(const nx_u32*)(p + 4);
    *lo = x; *hi = y;
}
static inline __attribute__((always_inline)) void nx_stp32(uint64_t a, uint64_t v0, uint64_t v1) {
    const uintptr_t p = NX_MEM_AT(a);
    *(nx_u32*)p = (uint32_t)v0; *(nx_u32*)(p + 4) = (uint32_t)v1;
}
#define recomp_ldp32(c, a, lo, hi) nx_ldp32(a, lo, hi)
#define recomp_ldp64(c, a, lo, hi) nx_ldp64(a, lo, hi)
#define recomp_stp32(c, a, v0, v1) nx_stp32(a, v0, v1)
#define recomp_stp64(c, a, v0, v1) nx_stp64(a, v0, v1)
)C";
    return h;
}

} // namespace nxrecomp::fastmem
