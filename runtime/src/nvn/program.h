// Programs (program.cpp), for the files that bind and translate shaders.
#pragma once

#include "nvn.h"

#include <array>
#include <optional>
#include <string>

namespace rt::nvn {

/// NVN's shader stages (the control section's +0x714 and the stage bits of
/// nvnCommandBufferBindProgram: bit = 1 << stage).
enum class Stage : u32 { kVertex = 0, kFragment, kGeometry, kTessControl, kTessEvaluation, kCompute };
constexpr u32 kStages = 6;

/// One stage of a program: what nvnProgramSetShaders keeps from its
/// NVNshaderData. The game may free the control section afterwards, so the
/// fields are copies. Layout of the control section (format 0x120/0xb, SDK
/// 8.2.0 and 12.3.2), measured on 1,579 shaders of Brothership.
struct ProgramStage {
    Stage stage{};
    u64 code = 0;             // GPU address of the stage's code memory (NVN's debug tag at +0)
    u32 program_offset = 0;   // +0x708: the SPH (graphics) or the instructions (compute) at code + this
    u32 program_size = 0;     // +0x6f8: SPH + instructions
    u32 constants_offset = 0; // +0x700: compiler constants (bound as c1) at code + this
    u32 constants_size = 0;   // +0x6fc
    u32 registers = 0;        // +0x70c (likely)
    u32 local_memory = 0;     // +0x710 (likely)
    u32 workgroup[3] = {};    // +0x748.. (compute)
    u32 shared_memory = 0;    // +0x754 (compute, likely)
    u64 tag_word = 0;         // +0x778: the debug tag's last word (mostly 0; a hash otherwise)
};

/// The bytes of a control section the parser reads.
constexpr u32 kControlSize = 0x800;

/// Reads the control section at `control` (guest) for code at `code`; nullopt
/// with the reason for a format the layer does not know.
std::optional<ProgramStage> ParseControl(u64 control, u64 code, std::string& why);
/// The same on a copy of the control section (kControlSize bytes).
std::optional<ProgramStage> ParseControlBytes(const u8* control, u64 code, std::string& why);

struct Program : Object {
    static constexpr Type kType = Type::kProgram;
    std::array<std::optional<ProgramStage>, kStages> stages; // by Stage
    u32 stage_mask = 0;                                       // 1 << stage for each stage set
};

} // namespace rt::nvn
