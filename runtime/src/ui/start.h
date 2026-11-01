// The start screen (LAUNCHER_PLAN Phase 1): shown in the window before the game boots - Play, Settings, Quit. The
// settings page is the options table (options.h) and the mods folder (mods.h); changes go to the settings file at
// once and count from Play on (the window, the resolution and the mods are read when the game starts).
//
// Drawn with Dear ImGui (third_party/imgui) on the NVN layer's Vulkan device (nvn/display.h ShowUi); mouse, keyboard
// and any SDL gamepad (the game's A / Enter confirms, B / Escape goes back: the button layout setting
// decides which buttons those are), so it works on the Steam Deck.
#pragma once

#include <string>
#include <vector>

namespace rt::ui {

/// Whether this run shows the start screen: a window is open, the NVN layer draws (the driver seam has no start
/// screen), and nothing asked to skip it - `--play` on the command line, RT_NO_LAUNCHER=1, or scripted input
/// (RT_INPUT, the bench scripts' and tests' runs).
bool Wanted(int argc, char** argv);

/// Runs the start screen on the main thread until the player chooses Play (RT_UI_TEST=play@<seconds>: by itself
/// after that long, a test; RT_UI_TEST=page=settings|mods opens on that page); Quit or closing the window ends the
/// process. `built_mods`: the mods lifted into this build (nxrecomp --mod), by name. `player`: a player start - the
/// game is built from the start screen (build.h: the Build page; Play goes there while the build folder's module is
/// not current) and loaded after Run returns. RT_UI_TEST also takes page=build and build (press Build at once); its
/// play@ waits for a running build and a current module.
void Run(const std::vector<std::string>& built_mods, bool player);

} // namespace rt::ui
