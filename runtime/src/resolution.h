// The internal resolution of the run (RESOLUTION_PLAN Phase 1): what the game's screen-sized render targets are
// drawn at, as a factor on the game's own handheld 1280x720.
//
// RT_RES=360p|540p|720p|1080p|1440p|2160p|auto (4k = 2160p). The factors are yuzu's (up_scale / 2^down_shift, the
// form drippu's rescaling pass takes): 0.5, 0.75, 1, 1.5, 2, 3. auto follows the window: the largest of 720p,
// 1080p and 1440p that fits its pixel height (the Steam Deck's 1280x800: 720p; headless: 720p) - 4K and the
// factors under 1 only when named (PORT_PLAN §6.1). Unset: 720p. The NVN layer scales (images.cpp ScalesWith).
#pragma once

#include "rt.h"

namespace rt::resolution {

struct Scale {
    u32 up_scale = 1, down_shift = 0; // factor = up_scale / 2^down_shift
    u32 width = 1280, height = 720;   // the game's 1280x720 at that factor
    float factor = 1.0f;
    bool Scaled() const { return up_scale != 1 || down_shift != 0; }
};

/// Chooses the run's scale from RT_RES and the window's height in pixels (0: no window). window::Open calls it
/// once; stops the run for an unknown value.
void Choose(int window_height_px);

/// The run's scale (720p if Choose was not called: tools, tests).
const Scale& Get();

} // namespace rt::resolution
