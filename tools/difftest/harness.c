/* Shared-library entry point that runs one generated test function on a flat
   CPU state, so difftest.py can drive it through ctypes. */
#include <fenv.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "recomp_runtime.h"

typedef void (*TestFn)(GuestContext*);
extern const TestFn g_tests[];
extern const unsigned char g_test_unhandled[];
extern const unsigned char g_test_terminator[];
extern const unsigned char g_test_simd[];
extern const int g_test_count;

/* Only block dispatch uses this, and single-instruction tests never dispatch. */
BlockFn recomp_lookup(uint64_t pc) {
    (void)pc;
    return 0;
}

/* Mirrored by DtState in difftest.py. x[31] is SP; nzcv uses PSTATE bits 31..28. */
typedef struct DtState {
    uint64_t x[32];
    uint64_t pc;
    uint64_t nzcv;
    uint64_t v[32][2];
    uint64_t fpcr;
    uint64_t fpsr;
} DtState;

static GuestContext g_ctx;

/* dt_set_fastmem(1): the tests get a host page table over the memory window, laid out as the runtime's
   (4 KB pages, one pointer-sized entry: host address of guest 0 | tag), so loads and stores take the page
   table paths (nx_mem.h in --inline-mem builds, drippu's helpers otherwise). Mode 2 tags every other
   page GPU-tracked: those accesses go through the load/store callbacks below, which read and write the
   same window. Mode 0: no page table (the flat window), as before. */
enum { kTagMemory = 0, kTagGpu = 2, kTagMask = 7 };
static int g_fastmem;
static RecompHostMem g_hm;
static uintptr_t* g_pages;
static uint64_t g_pages_n;
static uint8_t* g_win;
static uint64_t g_win_base, g_win_size;

static uint8_t* WinPtr(uint64_t va, uint32_t size) {
    va &= 0xffffffffffffULL;
    if (va < g_win_base || va - g_win_base > g_win_size || size > g_win_size - (va - g_win_base)) return 0;
    return g_win + (va - g_win_base);
}
static uint64_t CbLoad(void* user, uint64_t va, uint32_t size) {
    (void)user;
    uint64_t v = 0;
    uint8_t* p = WinPtr(va, size);
    if (p) memcpy(&v, p, size);
    return v;
}
static void CbStore(void* user, uint64_t va, uint32_t size, uint64_t v) {
    (void)user;
    uint8_t* p = WinPtr(va, size);
    if (p) memcpy(p, &v, size);
}

void dt_set_fastmem(int mode) { g_fastmem = mode; }

/* --direct-mem tests (lifter/src/fastmem.cpp DirectHeader) access guest memory at this host address plus the
   guest address, as the runtime's window (rt.h kHostWindowBase). dt_direct_window() reserves the whole 39-bit
   space there, readable and writable everywhere (pages appear on first touch, so a wild address reads zero
   as the callbacks' window does); difftest.py copies a module's memory in and passes the window to dt_run. */
#define DT_DIRECT_BASE 0x400000000000ULL
void* dt_direct_window(void) {
    static void* w;
    if (!w) {
        void* p = mmap((void*)DT_DIRECT_BASE, 1ULL << 39, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) return 0;
        if (p != (void*)DT_DIRECT_BASE) {
            munmap(p, 1ULL << 39);
            return 0;
        }
        w = p;
    }
    return w;
}

static void SetUpPages(uint8_t* mem, uint64_t base, uint64_t size) {
    const uint64_t end = base + size, n = (end + 0xFFF) >> 12;
    if (n > g_pages_n) {
        free(g_pages);
        g_pages = calloc(n, sizeof *g_pages);
        g_pages_n = n;
    } else {
        memset(g_pages, 0, n * sizeof *g_pages);
    }
    const uintptr_t host = (uintptr_t)mem - (uintptr_t)base;
    if (host & kTagMask) abort(); /* the tag bits must be free */
    for (uint64_t pg = base >> 12; pg < end >> 12; ++pg) {
        g_pages[pg] = host | (g_fastmem == 2 && (pg & 1) ? kTagGpu : kTagMemory);
    }
    g_win = mem, g_win_base = base, g_win_size = size;
    memset(&g_hm, 0, sizeof g_hm);
    g_hm.load = CbLoad;
    g_hm.store = CbStore;
    g_hm.page_entries = g_pages;
    g_hm.page_entry_stride = sizeof(uintptr_t);
    g_hm.page_bits = 12;
    g_hm.pointer_mask = ~(uint64_t)kTagMask;
    g_hm.address_space_max = n << 12;
    g_hm.page_type_mask = kTagMask;
    g_hm.page_type_memory = kTagMemory;
    g_hm.page_type_rasterizer_cached = kTagGpu;
    g_hm.page_type_debug = 1;
}

/* Incremented by fpx_probe.h in --fpx builds; stays zero otherwise. */
unsigned long long dt_fpx_probe_counts[3];
unsigned long long dt_fpx_probe(int k) { return dt_fpx_probe_counts[k]; }

int dt_count(void) { return g_test_count; }
int dt_unhandled(int idx) { return g_test_unhandled[idx]; }
int dt_simd(int idx) { return g_test_simd[idx]; } /* from the lifter's SIMD emitter (--simd) */

/* Returns the context's halted code: 0 when the instruction ran to completion. */
int dt_run(int idx, DtState* s, uint8_t* mem, uint64_t mem_base, uint64_t mem_size) {
    GuestContext* c = &g_ctx;
    memset(c, 0, sizeof *c);
    memcpy(c->x, s->x, sizeof c->x);
    c->pc = s->pc;
    c->n = (s->nzcv >> 31) & 1;
    c->z = (s->nzcv >> 30) & 1;
    c->c = (s->nzcv >> 29) & 1;
    c->v = (s->nzcv >> 28) & 1;
    memcpy(c->vreg, s->v, sizeof c->vreg);
    c->fpcr = s->fpcr;
    c->fpsr = s->fpsr;
    c->mem = mem;
    c->mem_base_vaddr = mem_base;
    c->mem_size = mem_size;
    c->pending_svc = ~0ULL;
    if (g_fastmem) {
        static uint8_t* set_mem;
        static uint64_t set_base, set_size;
        static int set_mode;
        if (mem != set_mem || mem_base != set_base || mem_size != set_size || g_fastmem != set_mode) {
            SetUpPages(mem, mem_base, mem_size);
            set_mem = mem, set_base = mem_base, set_size = mem_size, set_mode = g_fastmem;
        }
        c->host_mem = &g_hm;
    }

    /* Host FP state must not leak between trials. */
    fesetenv(FE_DFL_ENV);
    g_tests[idx](c);
    fesetenv(FE_DFL_ENV);

    memcpy(s->x, c->x, sizeof s->x);
    if (!g_test_terminator[idx] && c->pc == s->pc) {
        c->pc += 4;
    }
    s->pc = c->pc;
    s->nzcv = ((uint64_t)(c->n & 1) << 31) | ((uint64_t)(c->z & 1) << 30) |
              ((uint64_t)(c->c & 1) << 29) | ((uint64_t)(c->v & 1) << 28);
    memcpy(s->v, c->vreg, sizeof s->v);
    s->fpcr = c->fpcr;
    s->fpsr = c->fpsr;
    return c->halted;
}
