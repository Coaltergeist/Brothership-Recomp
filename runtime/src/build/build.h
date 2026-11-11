// The game builder: builds the game module (game/game.h) on the player's machine from their
// own dump - the lifter (nxrecomp) on the game's exefs, the lifted C compiled with the bundled clang, the objects
// linked with lld - with progress, an estimate, cancel and a log. Then the shader seed: the
// game's shader stages extracted from its archives (seed/extract.h) and translated into the pipeline cache and shader
// store by this program run headless (RT_NVN_SEED_ONLY); the game plays without it, with first-use stutter. Started by `nxrt --build` or the start screen's
// Build page; tools/build_game.sh is the reference for the commands (flags.h).
//
// Where things are:
//   the game files   RT_GAME_DIR (a player option): a folder of .nsp files (the base game + its update, unpacked with
//                    RT_KEYS - the keys folder - into <build folder>/game: unpack.h), or an unpacked dump: exefs/
//                    (rtld, main, subsdk*, sdk, main.npdm, compressed as dumped or hactool's --uncompressed copies)
//                    and romfs/
//   the build folder RT_BUILD_DIR (a player option), default %LOCALAPPDATA%\brothership-recomp\build (Windows),
//                    $XDG_DATA_HOME (~/.local/share)/brothership-recomp/build: gen/ (the lift: C, image.bin,
//                    symbols.txt), obj/ (objects, kept: an update compiles only changed files), the module
//                    (nxgame.dll / libnxgame.so), nxgame.id, timings.txt, build.log
//   the tools        RT_BUILD_TOOLS, else tools/ beside the executable (tools/make_toolchain.sh fills it: nxrecomp,
//                    llvm/ with clang + the linker + clang's headers, game/ with the module's own sources, mods/
//                    with the mods that ship); a Linux source checkout falls back to the repository's own
//
// The module id (NxGame.id, also in nxgame.id beside the module): a hash of the tools (NXRT_TOOLCHAIN_ID: the lifter's
// and the module's sources, drippu's translator, flags.h), the game's modules by build id and the exefs mods lifted
// in. A module whose id is not the one this runtime would build now is rebuilt, not run (LASRecomp's GAMELIB_ID).
#pragma once

#include <string>
#include <vector>

namespace rt::build {

/// The player's paths (from the options).
struct Paths {
    std::string source, keys;            // RT_GAME_DIR as chosen (an unpacked dump, or a folder of .nsp files), RT_KEYS
    bool packed = false;                 // `source` holds .nsp files: unpacked into <dir>/game (unpack.h)
    std::string game, dir;               // the unpacked game (may be empty: not chosen yet), RT_BUILD_DIR or its default
    std::string gen, obj, module, id_file, log;
    std::string image, symbols;          // the lift's image.bin and symbols.txt (the runtime's arguments)
    std::string romfs, npdm;             // the game's romfs/ and exefs/main.npdm (RT_ROMFS, RT_NPDM's defaults)
};
Paths Where();

/// RT_BUILD_DIR's default.
std::string DefaultDir();

/// The id a module built now would have; "" when it cannot be built (why: the game files or tools are missing).
std::string WantedId(std::string* why);

/// Whether the build folder holds a finished module with WantedId (why: what is missing or changed, for the player).
bool Ready(std::string* why);

/// The exefs mods lifted into the build folder's module, by name (from nxgame.id; before the module is loaded).
std::vector<std::string> BuiltMods();

enum class Stage { kIdle, kUnpack, kLift, kCompile, kLink, kExtract, kSeed, kSeedNext, kDone, kFailed, kCancelled };

struct Status {
    Stage stage = Stage::kIdle;
    int done = 0, total = 0, jobs = 0; // sources compiled / to compile this run (kSeed: stages), compile jobs
    double elapsed = 0;                // seconds since the start
    double remaining = -1;             // estimated seconds left; -1: no estimate yet
    std::string step;                  // what it is doing, short
    std::string problem;               // kFailed: what went wrong, for the player (the log has the rest)
};

/// Which steps a Start runs: all of them (`nxrt --build`); the game's - unpack, lift, compile, link - ending in
/// Stage::kSeedNext (the start screen, which then hands the shader steps to `nxrt --seed` and ends); or the shader
/// steps - extract, seed - alone (`nxrt --seed`). The seed never runs beside a start screen: a seed made while one
/// was open (its own Vulkan device) left most of the stages out of the pipeline cache.
enum class Steps { kAll, kGame, kShaders };

/// Starts a build on a thread of its own; false when one is running. Jobs: RT_BUILD_JOBS, else half the logical
/// cores (the heat rule), capped by the free memory (kJobMemory each) - at least one.
bool Start(Steps steps = Steps::kAll);
Status Get();
/// Ends a running build's processes; Get() then says kCancelled. Objects already compiled are kept.
void Cancel();
/// Whether a build is running; Stop: Cancel and wait (up to 10 s) for its processes to end (quitting).
bool Running();
void Stop();
const char* StageName(Stage s);

/// `nxrt --build`: builds with the settings' paths, progress on stdout, Ctrl+C cancels; the process's exit code. A
/// current module with no seed marker beside the pipeline cache gets the shader steps alone. `launcher` (`--prepare`):
/// each status line is followed by `progress <0-1> <step>`, the form RecompLauncher reads.
int Command(bool launcher = false);

/// Whether a seed marker (`<program id>.nvn.seed`) is beside the pipeline cache (RT_PIPELINE_CACHE, else the per-user
/// folder's cache\): the seed ran for some driver (the seed run itself checks that it was this one).
bool SeedMarked();

} // namespace rt::build
