// The player's mods (PORT_PLAN §6 B2, B3): RT_MODS names the enabled ones (a player option), RT_MODS_DIR is the
// mods folder (default: options::UserDir()/mods), one folder per mod in the layout yuzu and our mods/ use: <name>/exefs/ (patches, lifted in with
// `nxrecomp --mod`: the build carries them, rt.cpp ApplyMods switches them) and <name>/romfs/ (files layered over
// the game's RomFS at run time, services/fs.cpp).
#pragma once

#include <string>
#include <vector>

namespace rt::mods {

/// Brothership's frame-rate patch (Fl4sh_#9174's 60FPS mod: the game's 30 fps limiter and its own vsync request
/// off), built in with nxrecomp --mod and switched by the Frame rate setting rather than by name: on at RT_FPS=60 or
/// uncapped (RT_FPS only caps; the game limits itself to 30 without it), off at 30. Not on the start screen's mods
/// page. Naming it in RT_MODS switches it on whatever RT_FPS says.
inline constexpr const char* kFrameRateMod = "[60FPS v1.0.1]";

/// The enabled mods, in RT_MODS order (a later one wins where two change the same thing); `all`: every mod of the
/// build (`built`, its exefs mods by name) and every folder of RT_MODS_DIR, by name. A name that is neither, or a
/// folder whose exefs/ the build lacks, stops the run (once, at the first call; later calls return the same list).
const std::vector<std::string>& Enabled(const std::vector<std::string>& built);

/// The mods folder, and its mod folders by name (the start screen lists them).
std::string Folder();
std::vector<std::string> Folders();
/// Whether mod folder `name` has romfs/ files, or exefs/ patches (which only work when lifted in: nxrecomp --mod).
bool HasRomfs(const std::string& name);
bool HasExefs(const std::string& name);

/// The romfs/ folders of the enabled mods that have one, in Enabled() order. After Enabled().
std::vector<std::string> RomfsLayers();

} // namespace rt::mods
