// The game module's interface (BUILD_PLAN Phase 1): what the runtime gives the lifted code (NxHost) and what the
// lifted code gives the runtime (NxGame), in one exported function, nx_game. The lifted code's own names - rt_jump,
// rt_svc, rt_break, recomp_lookup, rt_trace_pc - are defined beside it (entry.c) and forward to NxHost, so the
// generated files are the same in both builds:
//   NXRT_GAME=static - the lifted code linked into nxrt, nx_game called directly;
//   NXRT_GAME=module - the lifted code in its own library (nxgame.dll / libnxgame.so), loaded at start
//                      (runtime/src/game/load.cpp), so the runtime can be built and shipped without any game code.
// C and C++. A change to either struct's layout or meaning raises NX_GAME_INTERFACE: a module built for another
// interface is refused, not run. Within an interface, NxGame.id names what the module was built from (the builder,
// runtime/src/build/build.h): a module whose id differs from what the runtime would build now is rebuilt, not run.
#pragma once

#include <stdarg.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NX_GAME_INTERFACE 3

struct GuestContext;
struct RtFunc;
struct RtSegment;
struct NxModTable;

typedef void (*NxBlockFn)(struct GuestContext*);

/// The runtime's side, given to nx_game once before any lifted code runs.
typedef struct NxHost {
    uint32_t interface; // NX_GAME_INTERFACE
    uint32_t size;      // sizeof(NxHost)
    void (*svc)(struct GuestContext* c, uint32_t imm);
    void (*jump)(struct GuestContext* c, uint64_t target);
    void (*brk)(struct GuestContext* c, uint32_t imm, uint64_t pc);
    NxBlockFn (*lookup)(uint64_t pc);
    void (*print)(const char* fmt, va_list args); // the module's printf / fprintf (libc.c): to the runtime's log
} NxHost;

/// The lifted code's side (nxrecomp's nx_table.c and recomp_runtime.c).
typedef struct NxGame {
    uint32_t interface; // NX_GAME_INTERFACE
    uint32_t size;      // sizeof(NxGame)
    const struct RtFunc* funcs; // sorted by offset
    unsigned long func_count;
    const struct RtSegment* segments;
    unsigned long segment_count;
    uint64_t image_size, entry, load_base;
    unsigned long long direct_base;  // nxrecomp's direct memory: the host address guest memory is at; 0: checked
    const struct NxModTable* mods;   // nxrecomp --mod's table; NULL: none
    uint64_t* module_base;           // the lifted code's g_module_base (the runtime sets it)
    uint64_t* (*trace_pc)(void);     // the calling thread's rt_trace_pc (--trace-pc: where the thread is)
    const char* id;                  // what it was built from (build.h); "dev": not built by the builder
} NxGame;

/// The module's one export (entry.c).
const NxGame* nx_game(const NxHost* host);
typedef const NxGame* (*NxGameEntry)(const NxHost* host);
#define NX_GAME_ENTRY "nx_game"

#ifdef __cplusplus
}
#endif
