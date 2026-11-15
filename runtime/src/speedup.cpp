// Speed-up while a button is held (2026-11-12): testers use Fl4sh_#9174's "Increase gamespeed HOLD ZL" cheat
// (Atmosphere; useful for cutscenes and ship travel). The cheat rewrites the first instruction of
// AWorldSettings::GetEffectiveTimeDilation() (TimeDilation x MatineeTimeDilation x DemoPlayTimeDilation, the game's
// overall speed) to return 10 while ZL is held. A static recompile cannot rewrite the game's code while it runs, so
// the same is done natively: that function (reached only through its vtable) is overridden, and while the chosen
// button is held its result is multiplied by the chosen factor - the game's own slow motion and pauses keep their
// share. ZL is not the default: LuigiBattleBtn uses it to cancel.
//
//   RT_SPEEDUP         off (default) | 2x | 3x | 5x | 10x
//   RT_SPEEDUP_BUTTON  rstick (default) | lstick | zl | zr | l | r | minus   (the button still reaches the game too)
//
// Game 1.0.1's code (guest = main ELF + 0x8004000), checked word by word before anything is installed:
//   0xaf53be8  AWorldSettings::GetEffectiveTimeDilation(this) -> s0
#include "speedup.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "hle.h"
#include "input.h"
#include "options.h"
#include "rt.h"

namespace rt::speedup {
namespace {

constexpr u64 kTimeDilation = 0xaf53be8;
constexpr u32 kExpected[4] = {0xbd42e800, 0xbd42ec01, 0x1e210803, 0xbd42f001};

float g_factor = 1.0f;
u64 g_button = 0;
std::atomic<bool> g_told{false};

void TimeDilation(GuestContext* c, void*) {
    hle::CallOriginal(c, kTimeDilation);
    if ((input::Current().buttons & g_button) == 0) return;
    hle::RetF(c, hle::ArgF(c, 0) * g_factor);
    if (!g_told.exchange(true)) Log("speed-up: held (x%g)", g_factor);
}

} // namespace

void Install() {
    const char* speed = options::Get("RT_SPEEDUP");
    if (!speed || std::strcmp(speed, "off") == 0) return;
    g_factor = std::strtof(speed, nullptr);
    static const std::pair<const char*, u64> kButtons[] = {{"rstick", 1u << 5}, {"lstick", 1u << 4}, {"zl", 1u << 8},
                                                           {"zr", 1u << 9},     {"l", 1u << 6},      {"r", 1u << 7},
                                                           {"minus", 1u << 11}};
    const char* button = options::Get("RT_SPEEDUP_BUTTON");
    if (!button) button = "rstick";
    for (const auto& [name, bit] : kButtons) {
        if (std::strcmp(button, name) == 0) g_button = bit;
    }
    const Region* r = g_mem.Find(kTimeDilation);
    u32 words[4] = {};
    if (r && kTimeDilation + sizeof words <= r->base + r->size) std::memcpy(words, g_mem.Raw(kTimeDilation), sizeof words);
    if (!r || std::memcmp(words, kExpected, sizeof words) != 0) {
        Log("speed-up: off (the game's code at 0x%llx is not 1.0.1's)", (unsigned long long)kTimeDilation);
        return;
    }
    hle::OverrideGuestFunction("0xaf53be8", TimeDilation);
    Log("speed-up: on (x%g while %s is held; RT_SPEEDUP, RT_SPEEDUP_BUTTON)", g_factor, button);
}

} // namespace rt::speedup
