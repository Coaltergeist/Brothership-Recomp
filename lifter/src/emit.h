// C code generation: one C function per guest function.
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "analysis.h"

namespace nxrecomp {

struct EmitOptions {
    std::string out_dir;
    // RT_PC records the guest PC for crash reports: before every instruction
    // (kInsn, 3x compile cost), at function entry and after calls (kFunc), or not.
    enum class Trace { kOff, kFunc, kInsn } trace = Trace::kInsn;
    bool fpx = true;          // drippu's verified native-FP fast paths
    // FP/SIMD data-processing instructions as calls to one helper per encoding (fp_<word>, drippu's own
    // translation as its body, in funcs_fp_*.c) instead of the body inline in every function: the same
    // semantics, far smaller functions. The default since 2026-10-17
    // (--no-fp-helpers: inline, as before).
    bool fp_helpers = true;
    // Integer Advanced SIMD as host vector code inline (simd.cpp); what it does not cover
    // keeps drippu's translation (--no-simd: never used).
    bool simd = true;
    // Guest loads and stores: the common case inline (fastmem.cpp, nx_mem.h). Opt-in (--inline-mem) until the
    // load/store difftest passes; off: drippu's helpers at every access, as before.
    bool inline_mem = false;
    // Guest loads and stores as plain host accesses at the runtime's fixed window (fastmem.cpp DirectHeader).
    // The default since 2026-10-19 (measured); needs the NVN layer and
    // parallel threads. --no-direct-mem: the checked helpers, for RT_GIL=1, RT_NO_FASTMEM and the driver seam.
    bool direct_mem = true;
    size_t max_file_bytes = 4 << 20;
    u64 load_base = 0;        // linked program: the address its relocations assume (0 = relocatable)
    // Mods switched at run time (--mod): nx_table.c's nx_mod_table lists them with their writes
    // (program-relative), and each code word one changes is lifted twice, `if (nx_mod_on[k])` choosing.
    struct Mod {
        std::string name;
        std::vector<std::pair<u64, std::vector<u8>>> writes;
    };
    std::vector<Mod> mods;
    std::map<u64, std::vector<std::pair<size_t, u32>>> mod_words; // program offset -> (mod, its word), in mod order
};

struct EmitStats {
    size_t functions = 0, instructions = 0, unhandled = 0, files = 0;
    size_t direct_calls = 0, tail_calls = 0, indirect_jumps = 0, jump_tables = 0, svcs = 0;
    size_t plt_direct = 0; // PLT stubs whose GOT slot the static linker bound
    // fp_helpers: calls emitted, distinct helpers, and FP/SIMD bodies left inline (not self-contained, or
    // an encoding that translated differently at another address).
    size_t fp_calls = 0, fp_helpers = 0, fp_inline = 0;
    size_t fpx_sites = 0; // FP fast path inline (simd::EmitFp), its fp_ helper the fallback
    // simd: sites emitted as host vector code, and their distinct encodings.
    size_t simd_sites = 0, simd_encodings = 0;
    size_t mod_sites = 0; // code words lifted in two versions (EmitOptions::mods)
};

EmitStats Emit(const Image& img, const Analysis& a, const EmitOptions& opts);

// The lift's shared headers alone (recomp_runtime.h, nx_mem.h, nx_funcs.h), as Emit writes them: no game needed.
// The runtime built without game code (NXRT_GAME=module) needs only these (--headers-only).
void EmitHeaders(const EmitOptions& opts);

} // namespace nxrecomp
