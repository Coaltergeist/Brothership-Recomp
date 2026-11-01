// The start screen's look, shared with the shader cache's own run (seed.h): the ImGui style and the background
// picture (start.cpp).
#pragma once

#include <string>

struct ImTextureData;

namespace rt::ui {

/// The style both windows use (ImGui's dark one, our sizes and rounding); call after ImGui::CreateContext.
void ApplyStyle();

/// The background picture's file: RT_UI_BACKGROUND=<file> (any PNG), else background.png in the per-user folder (a
/// player's own), else background.png beside the program (the release's: art of our own only - never the game's,
/// which is Nintendo's); "" when there is none.
std::string BackgroundPath();

/// That picture as an ImGui texture (registered: delete it after ImGui::UnregisterUserTexture); nullptr: none.
ImTextureData* LoadBackground();

/// The picture filling the window (cropped to its aspect ratio, centred), darkened by `dim` (0-1); nothing without one.
void DrawBackground(ImTextureData* background, float dim);

} // namespace rt::ui
