/* nx_mem.h's fast path against a model of the memory map (difftest.py --inline-mem runs it): every load
   and store width and the pairs, at the last bytes of a page whose next page lies elsewhere in host
   memory (the fast path must refuse page-crossing accesses), on GPU-tracked, unmapped and out-of-range
   pages, with the top byte set (ignored, as on AArch64), and with access counters on (always the slow
   path). The slow path is drippu's helpers with callbacks that read the model. Exit status 0 = pass. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "recomp_runtime.h"
#include "nx_mem.h"

BlockFn recomp_lookup(uint64_t pc) {
    (void)pc;
    return 0;
}

enum { kPages = 8, kTagMemory = 0, kTagGpu = 2, kTagMask = 7 };
/* Guest pages 0..7: 1, 2, 4, 5 plain (each its own host buffer, so neighbours are not adjacent),
   3 GPU-tracked, 0, 6, 7 unmapped; the address space ends at page 7. */
static uint8_t* g_host[kPages];
static uintptr_t g_pages[kPages];
static uint8_t g_gpu[0x1000]; /* page 3's bytes (reached only through the callbacks) */
static RecompHostMem g_hm;
static int g_failures, g_checks;

static uint8_t* Byte(uint64_t va) {
    va &= 0xffffffffffffULL;
    const uint64_t pg = va >> 12;
    if (pg >= kPages) return 0;
    if (pg == 3) return g_gpu + (va & 0xFFF);
    return g_host[pg] ? g_host[pg] + (va & 0xFFF) : 0;
}
/* The model: byte by byte; unmapped bytes read 0 and ignore writes (as the runtime's callbacks would
   fault, the harness's ignore). */
static uint64_t CbLoad(void* u, uint64_t va, uint32_t size) {
    (void)u;
    uint64_t v = 0;
    for (uint32_t i = 0; i < size; ++i) {
        uint8_t* p = Byte(va + i);
        v |= (uint64_t)(p ? *p : 0) << (8 * i);
    }
    return v;
}
static void CbStore(void* u, uint64_t va, uint32_t size, uint64_t v) {
    (void)u;
    for (uint32_t i = 0; i < size; ++i) {
        uint8_t* p = Byte(va + i);
        if (p) *p = (uint8_t)(v >> (8 * i));
    }
}

static void Check(int ok, const char* what, uint64_t va, uint64_t got, uint64_t want) {
    ++g_checks;
    if (ok) return;
    if (++g_failures <= 20) {
        printf("FAIL %s at 0x%llx: got 0x%llx, want 0x%llx\n", what, (unsigned long long)va, (unsigned long long)got,
               (unsigned long long)want);
    }
}

static void Fill(void) {
    for (int pg = 0; pg < kPages; ++pg) {
        if (!g_host[pg]) continue;
        for (int i = 0; i < 0x1000; ++i) g_host[pg][i] = (uint8_t)(pg * 37 + i * 11 + 5);
    }
    for (int i = 0; i < 0x1000; ++i) g_gpu[i] = (uint8_t)(i * 13 + 1);
}

static void RunAt(GuestContext* c, uint64_t va) {
    static const int widths[] = {1, 2, 4, 8};
    for (int k = 0; k < 4; ++k) {
        const int w = widths[k];
        Fill();
        const uint64_t want = CbLoad(0, va, w);
        uint64_t got = w == 1 ? recomp_load8(c, va) : w == 2 ? recomp_load16(c, va) : w == 4 ? recomp_load32(c, va)
                                                                                           : recomp_load64(c, va);
        Check(got == want, "load", va, got, want);
        const uint64_t v = 0x8877665544332211ULL ^ va;
        if (w == 1) recomp_store8(c, va, v);
        else if (w == 2) recomp_store16(c, va, v);
        else if (w == 4) recomp_store32(c, va, v);
        else recomp_store64(c, va, v);
        const uint64_t mask = w == 8 ? ~0ULL : (1ULL << (8 * w)) - 1;
        uint64_t expect = 0;
        for (int i = 0; i < w; ++i) expect |= (uint64_t)(Byte(va + i) ? (uint8_t)(v >> (8 * i)) : 0) << (8 * i);
        got = CbLoad(0, va, w);
        Check(got == (expect & mask), "store", va, got, expect & mask);
    }
    for (int w = 4; w <= 8; w += 4) {
        Fill();
        uint64_t lo, hi, wlo = CbLoad(0, va, w), whi = CbLoad(0, va + w, w);
        if (w == 4) recomp_ldp32(c, va, &lo, &hi);
        else recomp_ldp64(c, va, &lo, &hi);
        Check(lo == wlo && hi == whi, w == 4 ? "ldp32" : "ldp64", va, lo ^ hi, wlo ^ whi);
        const uint64_t v0 = 0x0102030405060708ULL ^ va, v1 = 0x1112131415161718ULL ^ va;
        if (w == 4) recomp_stp32(c, va, v0, v1);
        else recomp_stp64(c, va, v0, v1);
        const uint64_t m = w == 8 ? ~0ULL : 0xFFFFFFFFULL;
        uint64_t e0 = 0, e1 = 0;
        for (int i = 0; i < w; ++i) {
            e0 |= (uint64_t)(Byte(va + i) ? (uint8_t)(v0 >> (8 * i)) : 0) << (8 * i);
            e1 |= (uint64_t)(Byte(va + w + i) ? (uint8_t)(v1 >> (8 * i)) : 0) << (8 * i);
        }
        Check(CbLoad(0, va, w) == (e0 & m) && CbLoad(0, va + w, w) == (e1 & m), w == 4 ? "stp32" : "stp64", va,
              CbLoad(0, va, w), e0 & m);
    }
}

int main(void) {
    for (int pg = 1; pg <= 5; ++pg) {
        if (pg == 3) continue;
        g_host[pg] = aligned_alloc(0x1000, 0x2000); /* separate allocations: neighbours not adjacent */
    }
    for (int pg = 0; pg < kPages; ++pg) {
        if (pg == 3) g_pages[pg] = kTagGpu; /* no pointer, as the runtime's GPU-tracked pages */
        else if (g_host[pg]) g_pages[pg] = ((uintptr_t)g_host[pg] - ((uintptr_t)pg << 12)) | kTagMemory;
    }
    g_hm.load = CbLoad;
    g_hm.store = CbStore;
    g_hm.page_entries = g_pages;
    g_hm.page_entry_stride = sizeof(uintptr_t);
    g_hm.page_bits = 12;
    g_hm.pointer_mask = ~(uint64_t)kTagMask;
    g_hm.address_space_max = (uint64_t)(kPages - 1) << 12; /* page 7 lies outside */
    g_hm.page_type_mask = kTagMask;
    g_hm.page_type_memory = kTagMemory;
    g_hm.page_type_rasterizer_cached = kTagGpu;
    g_hm.page_type_debug = 1;
    GuestContext ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.host_mem = &g_hm;
    uint64_t counters[16] = {0};
    for (int pass = 0; pass < 2; ++pass) {
        g_hm.counters = pass ? counters : 0;
        for (int pg = 0; pg < kPages; ++pg) {
            for (uint64_t off = 0; off < 0x1000; off += (off < 16 || off >= 0xFF0) ? 1 : 0x3F1) {
                RunAt(&ctx, ((uint64_t)pg << 12) + off);
                RunAt(&ctx, (0xAB00000000000000ULL) | (((uint64_t)pg << 12) + off)); /* top byte ignored */
            }
        }
    }
    printf("fastmem_test: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures != 0;
}
