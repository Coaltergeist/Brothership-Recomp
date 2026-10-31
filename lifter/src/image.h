// Guest module images: loading ELF/NRO/NSO into a flat, module-relative byte
// image, and reading the metadata the analysis needs (MOD0, .dynamic,
// relocations, .eh_frame_hdr, symbols).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nxrecomp {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using s32 = int32_t;
using s64 = int64_t;

struct Segment {
    u64 offset;    // module-relative address
    u64 file_size; // bytes with initial contents
    u64 mem_size;  // including zero fill
    bool exec;
    bool write;
};

struct Relocation {
    u64 offset;
    u32 type;
    u32 symbol;
    s64 addend;
};

struct DynSymbol {
    std::string name;
    u64 value;
    bool defined;
    bool function;
    u8 binding = 1;    // STB_LOCAL 0, STB_GLOBAL 1, STB_WEAK 2
    u8 visibility = 0; // STV_DEFAULT 0; anything else binds within the module
    bool absolute = false; // SHN_ABS: value is not module-relative
};

struct Image {
    std::string name;   // file stem, e.g. "main"
    std::string format; // "elf", "nro", "nso" or "linked"
    std::vector<u8> mem; // module bytes from offset 0, bss zero-filled
    std::vector<Segment> segments;
    u64 text_begin = 0, text_end = 0;
    u64 entry = 0;
    std::vector<u8> build_id; // NSO / NRO header (0x20 bytes); what exefs patches name (patch.h)

    // From MOD0 / .dynamic.
    u64 mod0 = 0, dynamic = 0, eh_frame_hdr_begin = 0, eh_frame_hdr_end = 0;
    std::vector<Relocation> relocations;
    std::vector<DynSymbol> dynsyms;
    std::vector<u64> fde_starts;
    std::vector<std::pair<u64, u64>> fde_ranges; // [begin, end) of every FDE that could be parsed
    u64 init = 0, fini = 0, init_array = 0, init_array_size = 0, fini_array = 0,
        fini_array_size = 0; // DT_INIT* / DT_FINI* (module-relative)

    // Linked programs only: GOT slots the static linker bound to a symbol
    // (slot offset -> target offset, both program-relative).
    std::map<u64, u64> bound_slots;

    // Function names (address -> name): ELF .symtab FUNC symbols, or for an NSO
    // its exported FUNC symbols (the only names it has).
    std::map<u64, std::string> symbols;

    u32 Word(u64 off) const;
    u64 Dword(u64 off) const;
    bool InText(u64 off) const;
    u64 TextEnd(u64 off) const; // end of the executable segment containing off, or 0
};

/// Loads an ELF (static PIE, as devkitA64 links homebrew), an NRO, or an
/// uncompressed NSO. Throws std::runtime_error on malformed input.
Image LoadImage(const std::string& path);

} // namespace nxrecomp
