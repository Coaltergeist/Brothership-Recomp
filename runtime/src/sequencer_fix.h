// Unreal Sequencer repeater events at frame rates above the sequences' time grid (sequencer_fix.cpp).
#pragma once

namespace rt::sequencer_fix {

/// Installs the fix if the game's code is 1.0.1's (else logs why not). After hle::Init, before guest code runs; only
/// when the frame-rate patch is on (at 30 fps every frame crosses a tick of the 30-tick sequences: nothing to fix).
void Install();

} // namespace rt::sequencer_fix
