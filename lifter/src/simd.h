// Integer Advanced SIMD (NEON) instructions as host vector code (SIMD_PLAN.md): GCC/Clang vector
// extensions, inline at each site, FPSR.QC from a lane mask. An encoding Emit() does not cover keeps
// drippu's translation (the fp_<word> helpers). Standalone (no lifter headers), so tools/difftest's
// generator compiles the same code.
#pragma once

#include <cstdint>
#include <string>

namespace nxrecomp::simd {

/// The C for one instruction, appended to `out`; false (and `out` untouched) for an encoding it does
/// not cover, reserved encodings included.
bool Emit(uint32_t insn, std::string& out);

/// FP fast path (SIMD_PLAN.md Phase 4): FADD/FSUB/FMUL/FDIV/FMAX/FMIN/FMAXNM/FMINNM (scalar S/D), FCMP/FCMPE,
/// and the vector .2S/.4S/.2D FADD/FSUB/FMUL (also by element)/FADDP/FABD/FDIV/FMAX/FMIN/FMAXNM/FMINNM,
/// FCMEQ/FCMGE/FCMGT/FACGE/FACGT (also against zero, FCMLE/FCMLT) and FRECPS/FRSQRTS (.2S/.4S), computed
/// natively on all lanes at once. Arithmetic is kept under drippu's FPX conditions (the gate RECOMP_FPX_OPEN,
/// every lane's result kept as drippu's per-lane fast path would); compares and max/min need only a default
/// FPCR and no NaN lane. Otherwise `fallback` (the exact translation, e.g. "fp_<word>(c);") runs. False (`out`
/// untouched) for an encoding it does not cover.
bool EmitFp(uint32_t insn, const std::string& fallback, std::string& out);

/// nx_simd.h: the vector types and helpers the emitted code uses.
const std::string& Header();

/// Every emitted block names an nxv_ type or helper: a file containing this includes nx_simd.h.
inline constexpr const char* kMarker = "nxv_";

} // namespace nxrecomp::simd
