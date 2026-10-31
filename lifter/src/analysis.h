// Function discovery: entry-point evidence ("seeds") and function ranges.
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "image.h"

namespace nxrecomp {

/// Where a function start came from. A start can carry several.
enum SeedSource : u32 {
    kSeedEntry = 1 << 0,      // module entry point
    kSeedExport = 1 << 1,     // defined FUNC in .dynsym
    kSeedReloc = 1 << 2,      // relocation target in .text (vtables, function pointers)
    kSeedCall = 1 << 3,       // direct BL target
    kSeedFde = 1 << 4,        // .eh_frame_hdr entry
    kSeedAddress = 1 << 5,    // ADR / ADRP+ADD materialised address in .text
    kSeedTailCall = 1 << 6,   // branch target outside its function's range
    kSeedExtra = 1 << 7,      // supplied by the user (e.g. a runtime gaps file)
    kSeedLinked = 1 << 8,     // a GOT slot the static linker bound here (references from other modules)
};

struct Function {
    u64 begin, end; // module-relative, [begin, end)
    u32 sources;
};

struct Analysis {
    std::map<u64, u32> seeds; // start -> SeedSource bits
    std::vector<Function> functions; // sorted by begin
    std::map<u64, std::vector<u64>> jump_tables; // BR offset -> case targets
    size_t unresolved_jump_tables = 0;
    std::vector<u64> unresolved_jump_table_sites; // BR offsets
    const Function* Find(u64 off) const; // function containing off
};

struct DiscoveryOptions {
    bool use_address_seeds = true;
    bool use_export_seeds = true; // off: measure the other evidence against the exports
    std::vector<u64> extra_seeds;
    std::vector<u64> linked_seeds;
};

Analysis Discover(const Image& img, const DiscoveryOptions& opts);

/// Compares discovered functions with the image's .symtab FUNC symbols.
std::string ValidateAgainstSymbols(const Image& img, const Analysis& a);

// Instruction decoding helpers shared with the emitter.
bool IsUnconditionalTransfer(u32 insn); // B, BR, RET: no fallthrough
bool DirectTarget(u32 insn, u64 pc, u64& target); // B, BL, B.cond, CBZ/CBNZ, TBZ/TBNZ

} // namespace nxrecomp
