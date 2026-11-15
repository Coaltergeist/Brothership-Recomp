// Behaviour-tree steps at the game's 30 fps pace above 30 fps (bt_pacing.cpp).
#pragma once

namespace rt::bt_pacing {

/// Installs the pacing if the game's code is 1.0.1's (else logs why not). After hle::Init, before guest code runs;
/// only when the frame-rate patch is on (at 30 fps every frame is one step: nothing to pace).
void Install();

} // namespace rt::bt_pacing
