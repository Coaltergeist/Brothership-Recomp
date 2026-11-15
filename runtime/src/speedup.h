// Speed-up while a button is held (speedup.cpp).
#pragma once

namespace rt::speedup {

/// Installs the speed-up if RT_SPEEDUP is not off and the game's code is 1.0.1's (else logs why not). After hle::Init,
/// before guest code runs.
void Install();

} // namespace rt::speedup
