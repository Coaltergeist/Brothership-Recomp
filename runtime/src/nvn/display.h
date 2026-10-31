// The NVN layer's image on the host window (NVN_PLAN 9.1 display, 9.3).
#pragma once

#include "texture.h"

namespace rt::nvn {

/// Shows texture `t` (the texture the game presented) in the host window, scaled to fit with black bars,
/// with the overlay's text box on top. Nothing without a window (RT_HEADLESS=1).
void ShowInWindow(const TextureState& t);

} // namespace rt::nvn
