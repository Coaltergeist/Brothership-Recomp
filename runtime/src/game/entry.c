// The game module's entry (game.h): built with the lifted code, in both builds. The names the lifted code calls are
// defined here and forward to the runtime's functions (NxHost); the lifted code's tables go back as NxGame.
#include "nx_funcs.h"

#include "game.h"

#if defined(_WIN32)
#define NX_EXPORT __declspec(dllexport)
#else
#define NX_EXPORT __attribute__((visibility("default")))
#endif

// Lifts that may lack them: nxrecomp's direct memory (nx_mem_direct_base) and --mod (nx_mod_table), defined in
// nx_table.c when used. Windows has no weak symbols: a stand-in by alternatename (as rt.cpp had them).
extern const struct NxModTable nx_mod_table;
#if defined(_WIN32)
const unsigned long long nx_mem_direct_base_none = 0;
const unsigned long long nx_mod_table_none = 0;
#pragma comment(linker, "/alternatename:nx_mem_direct_base=nx_mem_direct_base_none")
#pragma comment(linker, "/alternatename:nx_mod_table=nx_mod_table_none")
extern const unsigned long long nx_mem_direct_base;
#define NX_MODS ((const void*)&nx_mod_table == (const void*)&nx_mod_table_none ? 0 : &nx_mod_table)
#define NX_DIRECT_BASE nx_mem_direct_base
#else
extern __attribute__((weak)) const unsigned long long nx_mem_direct_base;
#pragma weak nx_mod_table
#define NX_MODS (&nx_mod_table)
#define NX_DIRECT_BASE (&nx_mem_direct_base ? nx_mem_direct_base : 0)
#endif

// The builder's id for this module (build.h: its obj/nx_game_id.h, given with -include); CMake's static build and
// tools/build_game.sh: "dev"
#ifndef NX_GAME_ID_STRING
#define NX_GAME_ID_STRING "dev"
#endif

_Thread_local uint64_t rt_trace_pc = 0;

const NxHost* g_nx_host; // (libc.c's printf too)

void rt_svc(GuestContext* c, uint32_t imm) { g_nx_host->svc(c, imm); }
void rt_jump(GuestContext* c, uint64_t target) { g_nx_host->jump(c, target); }
void rt_break(GuestContext* c, uint32_t imm, uint64_t pc) { g_nx_host->brk(c, imm, pc); }
BlockFn recomp_lookup(uint64_t pc) { return g_nx_host->lookup(pc); }

static uint64_t* TracePc(void) { return &rt_trace_pc; }

NX_EXPORT const NxGame* nx_game(const NxHost* host) {
    static NxGame game;
    if (!host || host->interface != NX_GAME_INTERFACE || host->size != sizeof(NxHost)) return 0;
    g_nx_host = host;
    game.interface = NX_GAME_INTERFACE;
    game.size = sizeof(NxGame);
    game.funcs = (const struct RtFunc*)rt_funcs;
    game.func_count = rt_func_count;
    game.segments = (const struct RtSegment*)rt_segments;
    game.segment_count = rt_segment_count;
    game.image_size = rt_image_size;
    game.entry = rt_entry;
    game.load_base = rt_load_base;
    game.direct_base = NX_DIRECT_BASE;
    game.mods = NX_MODS;
    game.module_base = &g_module_base;
    game.trace_pc = TracePc;
    game.id = NX_GAME_ID_STRING;
    return &game;
}
