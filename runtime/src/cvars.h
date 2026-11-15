// The game's Unreal console variables, set from the player's options before the game starts (cvars.cpp).
#pragma once

namespace rt::cvars {

/// Installs the setter if any option asks for a console variable and the game's code is 1.0.1's (else logs why
/// not). After hle::Init, before guest code runs.
void Install();

} // namespace rt::cvars
