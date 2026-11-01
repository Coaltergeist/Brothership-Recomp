// The player's dump unpacked: a folder of .nsp files (the base game and its update) and the
// player's keys -> exefs/ + romfs/ with the update applied, by hactool (third_party/hactool, the release's
// tools/hactool[.exe]). The base game's program NCA is cut out of its .nsp into a temporary file (hactool reads the
// update's RomFS patch against it: --basenca), the update's ticket gives its title key (rights-ID crypto: the 16
// bytes at +0x180 of the .tik, decrypted by hactool with the player's prod.keys). Never logs a key: hactool's output
// is kept only without lines that hold one.
#pragma once

#include <atomic>
#include <functional>
#include <string>

namespace rt::build {

/// Whether `source` (the player's Game files folder) holds .nsp files rather than an unpacked dump.
bool IsPacked(const std::string& source);

/// Whether `game` (the unpacked folder) is missing or was unpacked from other files than `source` holds now.
bool UnpackNeeded(const std::string& source, const std::string& game);

struct UnpackArgs {
    std::string source;  // the folder with the .nsp files
    std::string keys;    // the keys folder (prod.keys in it) or prod.keys itself
    std::string hactool; // the program
    std::string game;    // the unpacked game: <game>/exefs, <game>/romfs (made anew)
    std::string work;    // temporary files (the cut-out NCAs), removed after
    std::string log;     // hactool's output, without key lines
};

/// Unpacks; `progress` gets (fraction 0-1, step). False (`why`, for the player) on a problem or cancel.
bool Unpack(const UnpackArgs& a, const std::function<void(double, const char*)>& progress,
            const std::atomic<bool>& cancel, std::string* why);

} // namespace rt::build
