// Guest loads and stores inline (the memory path). drippu's helpers (recomp_load64 ...,
// in recomp_runtime.c) resolve every access through two more calls and their checks; at the title ->
// file-select transition they were ~65 % of GameThread's time (2026-10-18). nx_mem.h redefines the
// helper names as small inline functions that take the common case themselves: a plain memory page,
// the access within it, no access counters. Everything else calls drippu's helper, which decides as
// before. Standalone (no lifter headers), so tools/difftest's generator writes the same header.
#pragma once

#include <string>

namespace nxrecomp::fastmem {

/// nx_mem.h: include after recomp_runtime.h, in generated code only (recomp_runtime.c defines the
/// helpers the macros rename).
const std::string& Header();

/// Where the runtime places the guest address space in the host (runtime/src/rt.h kHostWindowBase): with
/// --direct-mem the generated code adds it to every guest address.
constexpr unsigned long long kDirectBase = 0x4000'0000'0000ULL;

/// nx_mem.h for --direct-mem: every load and store (pairs too) is one host access at kDirectBase plus the
/// guest address; the runtime keeps unmapped guest memory inaccessible, so a stray access faults.
/// Exclusives keep drippu's helpers. Include as Header().
const std::string& DirectHeader();

} // namespace nxrecomp::fastmem
