#include "image.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace nxrecomp {

namespace {

constexpr s64 DT_NULL = 0, DT_PLTRELSZ = 2, DT_STRTAB = 5, DT_SYMTAB = 6, DT_RELA = 7,
              DT_RELASZ = 8, DT_INIT = 12, DT_FINI = 13, DT_JMPREL = 23, DT_INIT_ARRAY = 25,
              DT_FINI_ARRAY = 26, DT_INIT_ARRAYSZ = 27, DT_FINI_ARRAYSZ = 28, DT_RELRSZ = 35,
              DT_RELR = 36;
constexpr u32 R_AARCH64_RELATIVE = 1027;

template <typename T>
T Read(const std::vector<u8>& b, u64 off) {
    if (off + sizeof(T) > b.size()) {
        throw std::runtime_error("read past end of image");
    }
    T v;
    std::memcpy(&v, b.data() + off, sizeof v);
    return v;
}

std::vector<u8> ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    return {std::istreambuf_iterator<char>(in), {}};
}

void Place(Image& img, const std::vector<u8>& file, u64 file_off, u64 mem_off, u64 file_size,
           u64 mem_size, bool exec, bool write) {
    if (file_off + file_size > file.size()) {
        throw std::runtime_error("segment runs past end of file");
    }
    if (img.mem.size() < mem_off + mem_size) {
        img.mem.resize(mem_off + mem_size, 0);
    }
    std::memcpy(img.mem.data() + mem_off, file.data() + file_off, file_size);
    img.segments.push_back({mem_off, file_size, mem_size, exec, write});
    if (exec) {
        img.text_begin = mem_off;
        img.text_end = mem_off + file_size;
    }
}

void LoadElf(Image& img, const std::vector<u8>& f) {
    img.format = "elf";
    if (f[4] != 2 || f[5] != 1 || Read<u16>(f, 0x12) != 183) {
        throw std::runtime_error("expected a little-endian ELF64 AArch64 file");
    }
    const u64 phoff = Read<u64>(f, 0x20), shoff = Read<u64>(f, 0x28);
    const u16 phentsize = Read<u16>(f, 0x36), phnum = Read<u16>(f, 0x38);
    const u16 shentsize = Read<u16>(f, 0x3A), shnum = Read<u16>(f, 0x3C);
    img.entry = Read<u64>(f, 0x18);
    for (u16 i = 0; i < phnum; ++i) {
        const u64 p = phoff + u64{i} * phentsize;
        if (Read<u32>(f, p) != 1) { // PT_LOAD
            continue;
        }
        const u32 flags = Read<u32>(f, p + 4);
        Place(img, f, Read<u64>(f, p + 8), Read<u64>(f, p + 16), Read<u64>(f, p + 32),
              Read<u64>(f, p + 40), flags & 1, flags & 2);
    }
    // .symtab function symbols, for naming and for validating discovery.
    for (u16 i = 0; i < shnum; ++i) {
        const u64 s = shoff + u64{i} * shentsize;
        if (Read<u32>(f, s + 4) != 2) { // SHT_SYMTAB
            continue;
        }
        const u64 off = Read<u64>(f, s + 24), size = Read<u64>(f, s + 32);
        const u32 link = Read<u32>(f, s + 40);
        const u64 str = Read<u64>(f, shoff + u64{link} * shentsize + 24);
        for (u64 e = off; e + 24 <= off + size; e += 24) {
            const u8 info = f[e + 4];
            const u16 shndx = Read<u16>(f, e + 6);
            if ((info & 0xF) == 2 && shndx != 0) { // STT_FUNC, defined
                const char* name = reinterpret_cast<const char*>(f.data() + str + Read<u32>(f, e));
                img.symbols.emplace(Read<u64>(f, e + 8), name);
            }
        }
    }
}

// NRO and NSO both describe text/rodata/data as (file offset, memory offset, size).
void LoadNro(Image& img, const std::vector<u8>& f) {
    img.format = "nro";
    for (int s = 0; s < 3; ++s) {
        const u32 off = Read<u32>(f, 0x20 + s * 8), size = Read<u32>(f, 0x24 + s * 8);
        const u64 bss = s == 2 ? Read<u32>(f, 0x38) : 0;
        Place(img, f, off, off, size, size + bss, s == 0, s == 2);
    }
    img.build_id.assign(f.begin() + 0x40, f.begin() + 0x60);
}

// An LZ4 block (an NSO's compressed segment) of exactly `size` bytes once decompressed.
std::vector<u8> Lz4Block(const u8* in, size_t in_size, size_t size) {
    std::vector<u8> out;
    out.reserve(size);
    const u8* const end = in + in_size;
    const auto length = [&](size_t n) { // a 4-bit length of 15 continues in bytes of 255
        if (n == 15) {
            for (u8 b = 255; b == 255 && in < end; n += b) b = *in++;
        }
        return n;
    };
    while (in < end) {
        const u8 token = *in++;
        const size_t lit = length(token >> 4);
        if (lit > size_t(end - in)) throw std::runtime_error("NSO: bad LZ4 data (literals)");
        out.insert(out.end(), in, in + lit);
        in += lit;
        if (in >= end) break; // the last sequence has literals only
        if (end - in < 2) throw std::runtime_error("NSO: bad LZ4 data (offset)");
        const size_t back = in[0] | in[1] << 8;
        in += 2;
        const size_t match = length(token & 15) + 4;
        if (back == 0 || back > out.size()) throw std::runtime_error("NSO: bad LZ4 data (match)");
        for (size_t i = 0, from = out.size() - back; i < match; ++i) out.push_back(out[from + i]); // (may overlap)
    }
    if (out.size() != size) throw std::runtime_error("NSO: a segment decompressed to the wrong size");
    return out;
}

// As the console's loader: compressed segments (flags bits 0-2, LZ4, the file's sizes at 0x60) decompressed, so the
// game's exefs files are read as dumped (hactool's --uncompressed copies read the same).
void LoadNso(Image& img, const std::vector<u8>& f) {
    img.format = "nso";
    const u32 flags = Read<u32>(f, 0xC);
    for (int s = 0; s < 3; ++s) {
        const u64 h = 0x10 + s * 0x10;
        const u32 off = Read<u32>(f, h), mem = Read<u32>(f, h + 4), size = Read<u32>(f, h + 8);
        const u64 bss = s == 2 ? Read<u32>(f, 0x3C) : 0;
        if (!(flags & (1u << s))) {
            Place(img, f, off, mem, size, size + bss, s == 0, s == 2);
            continue;
        }
        const u32 packed = Read<u32>(f, 0x60 + s * 4);
        if (u64{off} + packed > f.size()) throw std::runtime_error("NSO: a compressed segment runs past the file");
        const std::vector<u8> seg = Lz4Block(f.data() + off, packed, size);
        Place(img, seg, 0, mem, size, size + bss, s == 0, s == 2);
    }
    img.build_id.assign(f.begin() + 0x40, f.begin() + 0x60);
}

u64 Uleb(const Image& img, u64& at) {
    u64 v = 0;
    for (int shift = 0; at < img.mem.size(); shift += 7) {
        const u8 b = img.mem[at++];
        v |= u64{b & 0x7Fu} << shift;
        if (!(b & 0x80)) {
            break;
        }
    }
    return v;
}

std::string CString(const Image& img, u64 off) {
    std::string s;
    while (off < img.mem.size() && img.mem[off]) {
        s.push_back(static_cast<char>(img.mem[off++]));
    }
    return s;
}

// DWARF pointer encodings used by .eh_frame_hdr.
u64 ReadEncoded(const Image& img, u64& at, u8 enc, u64 datarel_base) {
    s64 v;
    switch (enc & 0x0F) {
    case 0x03: v = img.Word(at); at += 4; break;                       // udata4
    case 0x0B: v = static_cast<s32>(img.Word(at)); at += 4; break;     // sdata4
    case 0x04: case 0x0C: v = static_cast<s64>(img.Dword(at)); at += 8; break;
    default: throw std::runtime_error("unsupported .eh_frame_hdr pointer encoding");
    }
    switch (enc & 0x70) {
    case 0x10: v += static_cast<s64>(at - ((enc & 0x0F) == 0x04 || (enc & 0x0F) == 0x0C ? 8 : 4)); break;
    case 0x30: v += static_cast<s64>(datarel_base); break;
    default: break;
    }
    return static_cast<u64>(v);
}

// pc_range of the FDE at `fde`, using the pointer encoding from its CIE's
// "zR" augmentation. False for anything unusual (64-bit lengths, no 'z').
bool FdeRange(const Image& img, u64 fde, u64& range) {
    if (fde + 8 > img.mem.size() || img.Word(fde) == 0xFFFFFFFF) {
        return false;
    }
    const u64 cie = fde + 4 - img.Word(fde + 4);
    if (cie + 9 > img.mem.size()) {
        return false;
    }
    u64 at = cie + 8;
    const u8 version = img.mem[at++];
    const std::string aug = CString(img, at);
    at += aug.size() + 1;
    if (aug.empty() || aug[0] != 'z') {
        return false;
    }
    Uleb(img, at); // code alignment
    Uleb(img, at); // data alignment (SLEB; only skipped)
    version == 1 ? (void)at++ : (void)Uleb(img, at); // return address register
    Uleb(img, at); // augmentation data length
    u8 enc = 0;
    for (size_t k = 1; k < aug.size(); ++k) {
        if (aug[k] == 'R') {
            enc = img.mem[at++];
        } else if (aug[k] == 'L') {
            ++at;
        } else if (aug[k] == 'P') {
            const u8 penc = img.mem[at++];
            ReadEncoded(img, at, penc & 0x0F, 0);
        } else {
            return false;
        }
    }
    u64 p = fde + 8;
    ReadEncoded(img, p, enc, 0);      // pc_begin
    range = ReadEncoded(img, p, enc & 0x0F, 0); // pc_range: value only, no pc-relative part
    return true;
}

void ReadModuleMetadata(Image& img) {
    img.mod0 = img.text_begin + img.Word(img.text_begin + 4);
    if (img.mod0 + 0x1C > img.mem.size() || std::memcmp(&img.mem[img.mod0], "MOD0", 4) != 0) {
        return; // no MOD0: nothing more to read
    }
    auto rel = [&](u64 field) { return img.mod0 + static_cast<s32>(img.Word(img.mod0 + field)); };
    img.dynamic = rel(4);
    img.eh_frame_hdr_begin = rel(16);
    img.eh_frame_hdr_end = rel(20);

    std::map<s64, u64> dyn;
    for (u64 d = img.dynamic;; d += 16) {
        const s64 tag = static_cast<s64>(img.Dword(d));
        if (tag == DT_NULL) {
            break;
        }
        dyn.emplace(tag, img.Dword(d + 8));
    }
    if (dyn.count(DT_SYMTAB) && dyn.count(DT_STRTAB)) {
        // Symbol count: the symbol table conventionally ends where the string table starts.
        const u64 symtab = dyn[DT_SYMTAB], strtab = dyn[DT_STRTAB];
        for (u64 s = symtab; s + 24 <= strtab; s += 24) {
            const u8 info = img.mem[s + 4];
            const u16 shndx = static_cast<u16>(img.mem[s + 6] | (img.mem[s + 7] << 8));
            DynSymbol sym{CString(img, strtab + img.Word(s)), img.Dword(s + 8), shndx != 0,
                          (info & 0xF) == 2};
            sym.binding = info >> 4;
            sym.visibility = img.mem[s + 5] & 3;
            sym.absolute = shndx == 0xFFF1;
            img.dynsyms.push_back(std::move(sym));
        }
    }
    auto tag = [&](s64 t) { return dyn.count(t) ? dyn[t] : 0; };
    img.init = tag(DT_INIT);
    img.fini = tag(DT_FINI);
    img.init_array = tag(DT_INIT_ARRAY);
    img.init_array_size = tag(DT_INIT_ARRAYSZ);
    img.fini_array = tag(DT_FINI_ARRAY);
    img.fini_array_size = tag(DT_FINI_ARRAYSZ);
    for (auto [start_tag, size_tag] : {std::pair{DT_RELA, DT_RELASZ}, {DT_JMPREL, DT_PLTRELSZ}}) {
        if (!dyn.count(start_tag)) {
            continue;
        }
        const u64 start = dyn[start_tag], size = dyn[size_tag];
        for (u64 r = start; r + 24 <= start + size; r += 24) {
            const u64 info = img.Dword(r + 8);
            img.relocations.push_back({img.Dword(r), static_cast<u32>(info),
                                       static_cast<u32>(info >> 32),
                                       static_cast<s64>(img.Dword(r + 16))});
        }
    }
    // RELR (-z pack-relative-relocs, current libnx): relative relocations
    // whose addend is the word already stored at each location. An even entry
    // is an address; an odd entry is a bitmap of the 63 words after the last.
    if (dyn.count(DT_RELR)) {
        const u64 start = dyn[DT_RELR], size = dyn[DT_RELRSZ];
        u64 where = 0;
        auto add = [&](u64 at) {
            img.relocations.push_back({at, R_AARCH64_RELATIVE, 0, static_cast<s64>(img.Dword(at))});
        };
        for (u64 e = start; e + 8 <= start + size; e += 8) {
            const u64 entry = img.Dword(e);
            if ((entry & 1) == 0) {
                add(entry);
                where = entry + 8;
            } else {
                for (u64 bits = entry >> 1, at = where; bits; bits >>= 1, at += 8) {
                    if (bits & 1) {
                        add(at);
                    }
                }
                where += 63 * 8;
            }
        }
    }
    if (img.eh_frame_hdr_end > img.eh_frame_hdr_begin) {
        const u64 hdr = img.eh_frame_hdr_begin;
        const u8 ptr_enc = img.mem[hdr + 1], count_enc = img.mem[hdr + 2], table_enc = img.mem[hdr + 3];
        u64 at = hdr + 4;
        ReadEncoded(img, at, ptr_enc, hdr);
        const u64 count = ReadEncoded(img, at, count_enc, hdr);
        for (u64 i = 0; i < count; ++i) {
            img.fde_starts.push_back(ReadEncoded(img, at, table_enc, hdr));
            const u64 fde = ReadEncoded(img, at, table_enc, hdr);
            u64 range;
            bool ok;
            try {
                ok = FdeRange(img, fde, range);
            } catch (const std::runtime_error&) {
                ok = false; // an encoding ReadEncoded doesn't know
            }
            if (ok) {
                img.fde_ranges.emplace_back(img.fde_starts.back(), img.fde_starts.back() + range);
            }
        }
    }
}

} // namespace

u32 Image::Word(u64 off) const { return Read<u32>(mem, off); }
u64 Image::Dword(u64 off) const { return Read<u64>(mem, off); }

bool Image::InText(u64 off) const { return TextEnd(off) != 0; }

u64 Image::TextEnd(u64 off) const {
    for (const auto& s : segments) {
        if (s.exec && off >= s.offset && off < s.offset + s.file_size) {
            return s.offset + s.file_size;
        }
    }
    return 0;
}

Image LoadImage(const std::string& path) {
    const std::vector<u8> f = ReadFile(path);
    if (f.size() < 0x80) {
        throw std::runtime_error(path + ": file too small");
    }
    Image img;
    if (std::memcmp(f.data(), "\x7f" "ELF", 4) == 0) {
        LoadElf(img, f);
    } else if (std::memcmp(f.data() + 0x10, "NRO0", 4) == 0) {
        LoadNro(img, f);
    } else if (std::memcmp(f.data(), "NSO0", 4) == 0) {
        LoadNso(img, f);
    } else {
        throw std::runtime_error(path + ": not an ELF, NRO or NSO");
    }
    if (img.text_end == 0) {
        throw std::runtime_error(path + ": no executable segment");
    }
    img.name = std::filesystem::path(path).stem().string();
    ReadModuleMetadata(img);
    if (img.symbols.empty()) {
        for (const auto& s : img.dynsyms) {
            if (s.defined && s.function && !s.absolute && img.InText(s.value)) {
                img.symbols.emplace(s.value, s.name);
            }
        }
    }
    return img;
}

} // namespace nxrecomp
