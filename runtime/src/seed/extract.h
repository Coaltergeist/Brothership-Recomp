// The game's shader stages from its own files, for the seed (RELEASE_PLAN Phase 3; tools/shaderseed/shaderseed.py,
// the reference, did the same from retoc's 23 GB unpack). Unreal 4.27 ships every shader precompiled: each stage an
// FShaderMapResourceCode entry (int32 size, LZ4 bytes, int32 unpacked size, uint8 frequency) whose unpacked blob starts
// with "LSLGS" and holds the stage's NVN code memory, its 0x800-byte control section and its NVIDIA assembly text
// (old_plans/SHADER_PLAN.md §9). Material shaders are in the IoStore containers' chunks (.utoc + .ucas), the global
// ones in the .pak's GlobalShaderCache; both Oodle-compressed (seed/oodle.h). Read in place, chunk by chunk, on
// several threads: nothing but the stages is written.
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>

namespace rt::seed {

struct ExtractStats {
    size_t containers = 0, chunks = 0;  // IoStore containers and chunks read
    size_t entries = 0;                 // distinct shader entries found
    size_t stages = 0;                  // distinct stages written (<out>/<name>.code / .ctrl / .asm)
    size_t unsplit = 0;                 // entries that are no stage (no control section)
    size_t bad_chunks = 0;              // chunks whose unpacked bytes miss the TOC's SHA-1 (ooz: extract.cpp)
};

/// Every distinct stage of the game in `romfs` (all its .utoc / .ucas containers and .pak files) into `out` as
/// <16 hex digits of SHA-1(code + control)>.code / .ctrl / .asm - the names shaderseed.py gives. `progress` gets the
/// fraction of the containers' compressed bytes read so far; `cancel` stops it. False (`why`) on a problem.
bool ExtractStages(const std::string& romfs, const std::string& out, int threads,
                   const std::function<void(double)>& progress, const std::atomic<bool>& cancel, ExtractStats* stats,
                   std::string* why);

} // namespace rt::seed
