// The game's Unreal console variables (cvars), set from the player's options before the game starts (2026-11-12).
//
// A tester asked for the effects a Switch cheat file switches (film grain, vignette, auto exposure, chromatic
// aberration, lens flare, bloom level, FXAA, anisotropic filtering). The cheats write cvar values at 1.0.0 addresses;
// here the cvars are set by name through Unreal's console manager, as its console would: IConsoleManager::Get()
// .FindConsoleVariable(name)->Set(value, ECVF_SetByConsole). SetByConsole is the highest priority, so the device
// profile and scalability settings the engine applies later do not undo it (Unreal logs those as "ignored as it is
// lower priority"), and the variable's own change handling runs. The shipping build has no command-line way to set
// cvars (no -ini: override, ExecCmds or DPCVars; rom:/UE4CommandLine.txt is read but only takes plain switches).
//
// When: at nnMain's start - every cvar is registered by then (static constructors), and the engine has not started
// its render thread, so Set copies the value to both its game- and render-thread slots directly.
//
// Options -> cvars (only for values other than the default, which leaves the game's value alone):
//   RT_BLOOM            r.BloomQuality 4..0 (the game's 5 = "veryhigh")
//   RT_FILM_GRAIN off   r.Tonemapper.GrainQuantization 0, and r.Tonemapper.Quality without grain (3)
//   RT_VIGNETTE off     r.Tonemapper.Quality 1 (no vignette - and no grain: Unreal's levels add grain on top)
//   RT_AUTO_EXPOSURE    r.EyeAdaptationQuality 0
//   RT_CHROMATIC_ABERRATION  r.SceneColorFringeQuality 0
//   RT_LENS_FLARE       r.LensFlareQuality 0
//   RT_AA smaa / none   r.PostProcessAAQuality 0 (the game's FXAA off)
//   RT_ANISOTROPY n     r.MaxAnisotropy n
//   RT_CVARS            any `name=value;...` (the settings file; after the others, so it wins)
//
// Game 1.0.1's code (guest = main ELF + 0x8004000), checked word by word before anything is installed:
//   0xe49fce0  the GOT slot holding &IConsoleManager::Singleton; 0x8a76fac SetupSingleton (when it is still null)
//   0x9a99774  a static initializer's IConsoleManager::Get() (the pattern the addresses come from)
// vtables: IConsoleManager FindConsoleVariable +0xb0; IConsoleVariable IsVariableInt +0x40, AsVariableInt +0x60,
// Set +0x88 (from five cvar classes' vtables, rebuilt from main's relocations).
#include "cvars.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "hle.h"
#include "options.h"
#include "rt.h"

namespace rt::cvars {
namespace {

constexpr u64 kSingletonGot = 0xe49fce0;
constexpr u64 kSetupSingleton = 0x8a76fac;
constexpr u64 kFindConsoleVariable = 0xb0, kIsVariableInt = 0x40, kAsVariableInt = 0x60, kSet = 0x88;
constexpr u64 kSetByConsole = 0x09000000;
constexpr u64 kNameAt = kHostScratchRegion + 0x80000, kValueAt = kHostScratchRegion + 0x80400; // UTF-16 strings

struct Expect {
    u64 va;
    u32 words[4];
};
constexpr Expect kExpected[] = {
    {kSetupSingleton, {0x6dba23e9, 0xa9035ff8, 0x9002d577, 0xf9459ee8}},
    {0x9a99774, {0xd0025037, 0xf94672f7, 0xf94002e0, 0xb5000060}},
};

std::vector<std::pair<std::string, std::string>> g_wanted;
u64 g_nn_main = 0;

void WriteWide(u64 at, const std::string& s) {
    for (size_t i = 0; i <= s.size(); ++i) hle::WriteGuest<u16>(at + 2 * i, i < s.size() ? u8(s[i]) : 0);
}

void Want(const char* name, const std::string& value) {
    for (auto& w : g_wanted) {
        if (w.first == name) {
            w.second = value;
            return;
        }
    }
    g_wanted.emplace_back(name, value);
}

bool Is(const char* option, const char* value) {
    const char* v = options::Get(option);
    return v && std::strcmp(v, value) == 0;
}

void Choose() {
    if (const char* v = options::Get("RT_BLOOM")) {
        static const std::pair<const char*, const char*> kLevels[] = {
            {"high", "4"}, {"medium", "3"}, {"low", "2"}, {"verylow", "1"}, {"off", "0"}};
        for (const auto& [name, level] : kLevels) {
            if (std::strcmp(v, name) == 0) Want("r.BloomQuality", level);
        }
    }
    const bool grain_off = Is("RT_FILM_GRAIN", "off"), vignette_off = Is("RT_VIGNETTE", "off");
    if (grain_off) Want("r.Tonemapper.GrainQuantization", "0");
    if (vignette_off) Want("r.Tonemapper.Quality", "1");
    else if (grain_off) Want("r.Tonemapper.Quality", "3");
    if (Is("RT_AUTO_EXPOSURE", "off")) Want("r.EyeAdaptationQuality", "0");
    if (Is("RT_CHROMATIC_ABERRATION", "off")) Want("r.SceneColorFringeQuality", "0");
    if (Is("RT_LENS_FLARE", "off")) Want("r.LensFlareQuality", "0");
    if (Is("RT_AA", "smaa") || Is("RT_AA", "none")) Want("r.PostProcessAAQuality", "0");
    if (const char* v = options::Get("RT_ANISOTROPY"); v && std::strcmp(v, "1") != 0) Want("r.MaxAnisotropy", v);
    if (const char* list = options::Get("RT_CVARS")) {
        std::string item;
        for (const char* p = list;; ++p) {
            if (*p == ';' || *p == '\0') {
                const size_t eq = item.find('=');
                if (eq != std::string::npos && eq > 0) Want(item.substr(0, eq).c_str(), item.substr(eq + 1));
                else if (!item.empty()) Log("cvars: RT_CVARS item \"%s\" is not name=value: ignored", item.c_str());
                item.clear();
                if (*p == '\0') break;
            } else if (*p != ' ') {
                item.push_back(*p);
            }
        }
    }
}

void NnMain(GuestContext* c, void*) {
    u64 manager = hle::ReadGuest<u64>(hle::ReadGuest<u64>(kSingletonGot));
    if (!manager) {
        hle::CallGuest(c, kSetupSingleton);
        manager = hle::ReadGuest<u64>(hle::ReadGuest<u64>(kSingletonGot));
    }
    std::string done;
    for (const auto& [name, value] : g_wanted) {
        WriteWide(kNameAt, name);
        const u64 var = manager ? hle::CallGuest(c, hle::ReadGuest<u64>(hle::ReadGuest<u64>(manager) + kFindConsoleVariable),
                                                 {manager, kNameAt, 1})
                                : 0;
        if (!var) {
            Log("cvars: %s not found: not set", name.c_str());
            continue;
        }
        WriteWide(kValueAt, value);
        const u64 vtable = hle::ReadGuest<u64>(var);
        hle::CallGuest(c, hle::ReadGuest<u64>(vtable + kSet), {var, kValueAt, kSetByConsole});
        std::string now = "?";
        if (hle::CallGuest(c, hle::ReadGuest<u64>(vtable + kIsVariableInt), {var}) & 1) {
            const u64 data = hle::CallGuest(c, hle::ReadGuest<u64>(vtable + kAsVariableInt), {var});
            if (data) now = std::to_string(hle::ReadGuest<s32>(data));
        }
        done += " " + name + "=" + now;
        if (now != "?" && now != value) Log("cvars: %s=%s asked, the game kept %s", name.c_str(), value.c_str(), now.c_str());
    }
    Log("cvars: set before the game started:%s", done.c_str());
    hle::CallOriginal(c, g_nn_main);
}

} // namespace

void Install() {
    Choose();
    if (g_wanted.empty()) return;
    for (const Expect& e : kExpected) {
        const Region* r = g_mem.Find(e.va);
        u32 words[4] = {};
        if (r && e.va + sizeof words <= r->base + r->size) std::memcpy(words, g_mem.Raw(e.va), sizeof words);
        if (!r || std::memcmp(words, e.words, sizeof words) != 0) {
            Log("cvars: off (the game's code at 0x%llx is not 1.0.1's): the effect options do nothing",
                (unsigned long long)e.va);
            return;
        }
    }
    g_nn_main = hle::OverrideGuestFunction("nnMain", NnMain);
}

} // namespace rt::cvars
