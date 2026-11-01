// NVNprogram: programs and their shader stages.
//
// nvnProgramSetShaders gets one NVNshaderData per stage (16 bytes: the code's
// GPU address, the control section's CPU address). The layer keeps what the
// translation and the bindings need from the control section (program.h).
// Like NVN (SDK 8.2.0 and 12.3.2, `0x570624`), it writes a 24-byte debug tag
// into the game's code memory at the start of each stage's code when 0x30
// bytes or more are reserved there: {u32 magic 0x12345678 (compute
// 0x12345679), u32 program size, u64 program, u64 control +0x778} (the last
// word: 1,579 of 1,579 shaders of Brothership). Only debuggers read it.
//
// Tap mode: a shadow checks the parser and the tag rule against the real
// driver at every nvnProgramSetShaders ("nvn program model:" lines): the
// stage mask the driver keeps in the program (byte +0x68) and the tag it
// wrote into the code memory.
#include "program.h"

#include "tap.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <utility>

namespace rt::nvn {

namespace {

constexpr u32 kControlMagic = 0x98761234;
constexpr u32 kTagMagic = 0x12345678, kTagMagicCompute = 0x12345679;
constexpr u32 kTagSpace = 0x30; // reserved bytes before the program needed for the tag

struct DebugTag {
    u32 magic, program_size;
    u64 program, ext;
};
static_assert(sizeof(DebugTag) == 24);

DebugTag TagFor(const ProgramStage& s, u64 program) {
    return {s.stage == Stage::kCompute ? kTagMagicCompute : kTagMagic, s.program_size, program, s.tag_word};
}

u32 Word(const u8* control, u32 offset) {
    u32 v;
    std::memcpy(&v, control + offset, 4);
    return v;
}

} // namespace

std::optional<ProgramStage> ParseControl(u64 control, u64 code, std::string& why) {
    std::array<u8, kControlSize> bytes;
    g_mem.Read(control, bytes.data(), bytes.size());
    return ParseControlBytes(bytes.data(), code, why);
}

std::optional<ProgramStage> ParseControlBytes(const u8* control, u64 code, std::string& why) {
    char buf[160];
    const u32 magic = Word(control, 0x00), major = Word(control, 0x0c), minor = Word(control, 0x10);
    if (magic != kControlMagic) {
        std::snprintf(buf, sizeof buf, "control section magic 0x%x", magic);
        why = buf;
        return std::nullopt;
    }
    if (major != 0x120 || minor != 0xb) { // (SDK 3.5.0's 4/14 format)
        std::snprintf(buf, sizeof buf, "control section format 0x%x/0x%x (the layer knows 0x120/0xb)", major, minor);
        why = buf;
        return std::nullopt;
    }
    ProgramStage s;
    const u32 stage = Word(control, 0x714);
    if (stage >= kStages) {
        std::snprintf(buf, sizeof buf, "control section stage %u", stage);
        why = buf;
        return std::nullopt;
    }
    s.stage = static_cast<Stage>(stage);
    s.code = code;
    s.program_size = Word(control, 0x6f8);
    s.constants_size = Word(control, 0x6fc);
    s.constants_offset = Word(control, 0x700);
    s.program_offset = Word(control, 0x708);
    s.registers = Word(control, 0x70c);
    s.local_memory = Word(control, 0x710);
    std::memcpy(&s.tag_word, control + 0x778, 8);
    if (s.stage == Stage::kCompute) {
        for (u32 i = 0; i < 3; ++i) s.workgroup[i] = Word(control, 0x748 + 4 * i);
        s.shared_memory = Word(control, 0x754);
    }
    return s;
}

// --- Program ------------------------------------------------------------------------------------

NVN_PROC(nvnProgramInitialize, bool, u64 program, u64 device) {
    Create<Program>(program);
    return true; // as NVN, always
}
NVN_PROC(nvnProgramFinalize, void, u64 program) { Destroy<Program>(program); }
NVN_PROC(nvnProgramSetDebugLabel, void, u64 program, u64 label) { Get<Program>(program); }

NVN_PROC(nvnProgramSetShaders, bool, u64 program, s32 count, u64 data) {
    Program* o = Get<Program>(program);
    if (count < 1 || count > static_cast<s32>(kStages)) {
        Fatal("nvn: nvnProgramSetShaders: %d stages (the layer handles 1 to %u)", count, kStages);
    }
    o->stages = {};
    o->stage_mask = 0;
    for (s32 i = 0; i < count; ++i) {
        const u64 code = hle::ReadGuest<u64>(data + 16ULL * i), control = hle::ReadGuest<u64>(data + 16ULL * i + 8);
        std::string why;
        const auto s = ParseControl(control, code, why);
        if (!s) Fatal("nvn: nvnProgramSetShaders: stage %d: %s", i, why.c_str());
        if (!FindPool(code).pool) {
            Fatal("nvn: nvnProgramSetShaders: stage %d: code at 0x%llx is in no memory pool", i,
                  (unsigned long long)code);
        }
        if (s->program_offset >= kTagSpace) Store(code, TagFor(*s, program)); // our GPU address = guest address
        o->stages[static_cast<u32>(s->stage)] = *s;
        o->stage_mask |= 1u << static_cast<u32>(s->stage);
    }
    return true;
}

// --- Tap: the parser and the tag against the real driver ------------------------------------------

namespace {

void ShadowSetShaders(const std::string&, const GuestContext& before, const GuestContext& after) {
    static std::mutex mu;
    static u64 checked = 0, different = 0, unhandled = 0;
    static std::set<std::pair<u32, u32>> words; // (our mask, the driver's word +0x68) seen
    const u64 program = before.x[0], data = before.x[2];
    const s32 count = static_cast<s32>(before.x[1]);
    std::lock_guard lock(mu);
    ++checked;
    std::string problem;
    u32 mask = 0;
    for (s32 i = 0; i < count && i < static_cast<s32>(kStages); ++i) {
        const u64 code_gpu = hle::ReadGuest<u64>(data + 16ULL * i), control = hle::ReadGuest<u64>(data + 16ULL * i + 8);
        std::string why;
        const auto s = ParseControl(control, code_gpu, why);
        if (!s) {
            if (++unhandled <= 20) Log("nvn program model: not handled by the layer: %s", why.c_str());
            return;
        }
        mask |= 1u << static_cast<u32>(s->stage);
        const u64 code = DriverGpuToGuest(code_gpu);
        if (!code) {
            problem += " stage " + std::to_string(i) + ": code not in a known pool;";
            continue;
        }
        const auto tag = hle::ReadGuest<DebugTag>(code);
        const DebugTag want = TagFor(*s, program);
        const bool wrote = s->program_offset >= kTagSpace;
        if (wrote ? std::memcmp(&tag, &want, sizeof tag) != 0 : tag.magic == want.magic) {
            char buf[200];
            std::snprintf(buf, sizeof buf, " stage %d (offset 0x%x): tag {0x%x, 0x%x, 0x%llx, 0x%llx}, layer %s;", i,
                          s->program_offset, tag.magic, tag.program_size, (unsigned long long)tag.program,
                          (unsigned long long)tag.ext, wrote ? "writes one" : "writes none");
            problem += buf;
        }
    }
    const u32 word = hle::ReadGuest<u32>(program + 0x68);
    if ((word & 0xff) != mask) {
        char buf[64];
        std::snprintf(buf, sizeof buf, " stage mask: driver 0x%x, layer 0x%x;", word & 0xff, mask);
        problem += buf;
    }
    if (!(after.x[0] & 0xff)) problem += " the driver returned false;";
    if (!problem.empty() && ++different <= 20) Log("nvn program model: DIFFERENT:%s", problem.c_str());
    if (words.emplace(mask, word).second && words.size() <= 20) {
        Log("nvn program model: stage mask 0x%x, driver's program word +0x68 = 0x%x", mask, word);
    }
    u64 n = checked;
    while (n % 10 == 0) n /= 10;
    if (n == 1) {
        Log("nvn program model: %llu calls checked, %llu different, %llu not handled", (unsigned long long)checked,
            (unsigned long long)different, (unsigned long long)unhandled);
    }
}

} // namespace

void InstallProgramShadows() { tap::AddShadow("nvnProgramSetShaders", ShadowSetShaders); }

} // namespace rt::nvn
