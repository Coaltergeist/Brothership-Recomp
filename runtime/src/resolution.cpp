// The internal resolution of the run (resolution.h).
#include "resolution.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "services.h"
#include "options.h"

namespace rt::resolution {
namespace {

struct Choice {
    const char* name;
    u32 lines, up_scale, down_shift;
};
constexpr std::array<Choice, 6> kChoices{{
    {"360p", 360, 1, 1},
    {"540p", 540, 3, 2},
    {"720p", 720, 1, 0},
    {"1080p", 1080, 3, 1},
    {"1440p", 1440, 2, 0},
    {"2160p", 2160, 3, 0},
}};
constexpr u32 kAutoMaxLines = 1440; // auto never picks 4K (experimental) or under 720p

Scale g_scale;
std::string g_line; // Choose's choice, logged by LogChoice

Scale Make(const Choice& c) {
    Scale s;
    s.up_scale = c.up_scale, s.down_shift = c.down_shift;
    s.factor = static_cast<float>(c.up_scale) / static_cast<float>(1u << c.down_shift);
    s.width = 1280 * c.up_scale >> c.down_shift, s.height = 720 * c.up_scale >> c.down_shift;
    return s;
}

} // namespace

void Choose(int window_height_px) {
    const char* e = options::Get("RT_RES");
    const char* name = e ? e : "720p";
    if (std::strcmp(name, "4k") == 0 || std::strcmp(name, "4K") == 0) name = "2160p";
    const Choice* pick = nullptr;
    if (std::strcmp(name, "auto") == 0) {
        pick = &kChoices[2];
        for (const Choice& c : kChoices) {
            if (c.lines >= 720 && c.lines <= kAutoMaxLines && static_cast<int>(c.lines) <= window_height_px) pick = &c;
        }
    } else {
        for (const Choice& c : kChoices) {
            if (std::strcmp(name, c.name) == 0) pick = &c;
        }
        if (!pick) Fatal("resolution: RT_RES=%s (360p|540p|720p|1080p|1440p|2160p|auto)", e);
    }
    // A factor over 1 that is not a whole number (1080p's 1.5x) is drawn at the next whole one and the window's
    // blit shrinks it (display.cpp): at 1.5x one game pixel is 1.5 pixels, so the game's own one-pixel steps (fetch
    // offsets: +1 -> +1 but -1 -> -2; sizes from its constant buffers, which the rescaling pass never sees) land
    // between pixels - smeared textures and ragged outlines (a tester's 1080p shots, 2026-11-06; headless plaza
    // captures against 4K box-filtered to 1080p: 1.5x differs 5.0 per pixel, 2x shrunk 1.75). RT_RES_EXACT=1: as
    // named (A/B). Factors under 1 (540p) stay: they are for speed.
    Choice drawn = *pick;
    const char* exact = std::getenv("RT_RES_EXACT");
    const bool whole = drawn.up_scale % (1u << drawn.down_shift) == 0;
    const bool over_one = drawn.up_scale > (1u << drawn.down_shift);
    if (!whole && over_one && !(exact && *exact && *exact != '0')) {
        drawn.up_scale = (drawn.up_scale + (1u << drawn.down_shift) - 1) >> drawn.down_shift, drawn.down_shift = 0;
    }
    g_scale = Make(drawn);
    // RT_DOCKED's game renders 1600x900: the factors are for handheld's 720p
    if (ReportDocked() && g_scale.Scaled()) Fatal("resolution: RT_RES=%s with RT_DOCKED (factors are on 720p)", name);
    char window[48] = "no window";
    if (window_height_px) std::snprintf(window, sizeof window, "window %d pixels high", window_height_px);
    char shrunk[96] = "";
    if (drawn.up_scale != pick->up_scale) {
        std::snprintf(shrunk, sizeof shrunk, ", %s drawn at %ux and shrunk to the window (RT_RES_EXACT=1: %s as named)",
                      pick->name, drawn.up_scale, pick->name);
    }
    char line[512];
    std::snprintf(line, sizeof line, "resolution: internal %ux%u (%gx the game's 1280x720; RT_RES=%s, %s%s)%s",
                  g_scale.width, g_scale.height, g_scale.factor, e ? e : "unset", window, shrunk,
                  g_scale.Scaled() ? "; the NVN layer scales (RT_GPU=nvn)" : "");
    g_line = line;
}

void LogChoice() {
    if (!g_line.empty()) Log("%s", g_line.c_str());
}

const Scale& Get() {
    return g_scale;
}

} // namespace rt::resolution
