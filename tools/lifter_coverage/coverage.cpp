// Measures how much of an uncompressed NSO's .text drippu's AArch64 -> C lifter
// (third_party/drippu/src/core/recompiler/arm64_to_c.h) can translate.
//
// Usage: coverage <module.bin> <out.tsv>
//
// Input is an NSO0 file whose segments are stored uncompressed (hactool -t nso0
// --uncompressed). Prints a summary to stdout and writes every distinct
// unhandled encoding as "encoding<TAB>count<TAB>first_offset" to out.tsv.

#include "core/recompiler/arm64_to_c.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <unordered_map>

namespace {

struct Segment {
    uint32_t file_offset;
    uint32_t mem_offset;
    uint32_t size;
};

Segment ReadSegment(const std::vector<uint8_t>& f, size_t off) {
    Segment s;
    std::memcpy(&s, f.data() + off, sizeof s);
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <module.bin> <out.tsv>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)), {});
    if (file.size() < 0x100 || std::memcmp(file.data(), "NSO0", 4) != 0) {
        std::fprintf(stderr, "%s: not an NSO0 file\n", argv[1]);
        return 1;
    }
    uint32_t flags;
    std::memcpy(&flags, file.data() + 0xC, 4);
    if (flags & 7) {
        std::fprintf(stderr, "%s: segments are compressed (flags 0x%x)\n", argv[1], flags);
        return 1;
    }
    const Segment text = ReadSegment(file, 0x10);
    if (text.file_offset + static_cast<size_t>(text.size) > file.size()) {
        std::fprintf(stderr, "%s: .text runs past end of file\n", argv[1]);
        return 1;
    }
    const uint8_t* code = file.data() + text.file_offset;
    const uint64_t base = text.mem_offset;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(code);

    // A standalone recompilation has no JIT to fall back on, so enable the
    // forms drippu leaves to Dynarmic in hybrid mode.
    suyu::recomp::g_translate_all = true;

    const auto t0 = std::chrono::steady_clock::now();
    const auto blocks = suyu::recomp::DiscoverBlocks(code, text.size, base);

    uint64_t insns = 0, unhandled = 0, emitted_bytes = 0;
    std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> by_encoding;
    std::string out;
    for (const auto& b : blocks) {
        for (uint32_t k = 0; k < b.count; ++k) {
            const uint64_t pc = b.vaddr + uint64_t{k} * 4;
            const uint32_t insn = words[(pc - base) / 4];
            bool miss = false;
            out.clear();
            suyu::recomp::Translate(insn, pc, out, &miss);
            emitted_bytes += out.size();
            ++insns;
            if (miss) {
                ++unhandled;
                auto [it, fresh] = by_encoding.try_emplace(insn, 0, pc);
                ++it->second.first;
            }
        }
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::FILE* tsv = std::fopen(argv[2], "w");
    if (!tsv) {
        std::perror(argv[2]);
        return 1;
    }
    for (const auto& [enc, v] : by_encoding) {
        std::fprintf(tsv, "%08x\t%llu\t%llx\n", enc, static_cast<unsigned long long>(v.first),
                     static_cast<unsigned long long>(v.second));
    }
    std::fclose(tsv);

    std::printf("module          %s\n", argv[1]);
    std::printf(".text           0x%x bytes at 0x%llx\n", text.size,
                static_cast<unsigned long long>(base));
    std::printf("blocks          %zu\n", blocks.size());
    std::printf("instructions    %llu\n", static_cast<unsigned long long>(insns));
    std::printf("unhandled       %llu (%.3f%%), %zu distinct encodings\n",
                static_cast<unsigned long long>(unhandled),
                insns ? 100.0 * unhandled / insns : 0.0, by_encoding.size());
    std::printf("emitted C       %.1f MB (instruction bodies only)\n", emitted_bytes / 1e6);
    std::printf("time            %.1f s\n", secs);
    return 0;
}
