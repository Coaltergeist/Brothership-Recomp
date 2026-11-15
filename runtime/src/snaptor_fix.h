// Mega Snaptor's rush attack at 60 fps: no end-of-attack teleport offscreen (snaptor_fix.cpp).
#pragma once

namespace rt::snaptor_fix {

/// Installs the fix if the game's code is 1.0.1's (else logs why not). After hle::Init, before guest code runs;
/// only when the frame-rate patch is on (at 30 fps the game's own order brings the boss back).
void Install();

} // namespace rt::snaptor_fix
