// The internal resolution of the run: what the game's screen-sized render targets are
// drawn at, as a factor on the game's own handheld 1280x720.
//
// RT_RES=360p|540p|720p|1080p|1440p|2160p|auto (4k = 2160p). The factors are yuzu's (up_scale / 2^down_shift, the
// form drippu's rescaling pass takes): 0.5, 0.75, 1, 1.5, 2, 3. auto follows the window: the largest of 720p,
// 1080p and 1440p that fits its pixel height (the Steam Deck's 1280x800: 720p; headless: 720p) - 4K and the
// factors under 1 only when named. Unset: 720p. The NVN layer scales (images.cpp ScalesWith).
// 1080p's 1.5x is drawn at 2x and shrunk by the window's blit (resolution.cpp; RT_RES_EXACT=1: 1.5x).
#pragma once

#include "rt.h"

namespace rt::resolution {

struct Scale {
    u32 up_scale = 1, down_shift = 0; // factor = up_scale / 2^down_shift
    u32 width = 1280, height = 720;   // the game's 1280x720 at that factor
    float factor = 1.0f;
    bool Scaled() const { return up_scale != 1 || down_shift != 0; }
};

/// Chooses the run's scale from RT_RES and the window's height in pixels (0: no window). window::Open and
/// window::Reapply (after the start screen) call it; stops the run for an unknown value.
void Choose(int window_height_px);

/// Logs the last choice. rt.cpp calls it once the start screen (which may change RT_RES and the window) is done: a
/// line from before it named a resolution the game never used and read as a bug (a tester's 1.4 log, 2026-11-06).
void LogChoice();

/// The run's scale (720p if Choose was not called: tools, tests).
const Scale& Get();

} // namespace rt::resolution
