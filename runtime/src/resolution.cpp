// The internal resolution of the run (resolution.h, RESOLUTION_PLAN Phase 1).
#include "resolution.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
    g_scale = Make(*pick);
    // RT_DOCKED's game renders 1600x900: the factors are for handheld's 720p
    if (ReportDocked() && g_scale.Scaled()) Fatal("resolution: RT_RES=%s with RT_DOCKED (factors are on 720p)", name);
    char window[48] = "no window";
    if (window_height_px) std::snprintf(window, sizeof window, "window %d pixels high", window_height_px);
    Log("resolution: internal %ux%u (%gx the game's 1280x720; RT_RES=%s, %s)%s", g_scale.width, g_scale.height,
        g_scale.factor, e ? e : "unset", window, g_scale.Scaled() ? "; the NVN layer scales (RT_GPU=nvn)" : "");
}

const Scale& Get() {
    return g_scale;
}

} // namespace rt::resolution
