// The lifted code's tables (game.h): linked in, or loaded from the game module (NXRT_GAME_MODULE).
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <SDL3/SDL.h>

#include "../rt.h"

namespace rt {
namespace {

const NxGame* g_game = nullptr;

/// The module's printf / fprintf (game/libc.c: drippu's messages, e.g. an unhandled instruction) into the log.
void Print(const char* fmt, va_list args) {
    std::vfprintf(stderr, fmt, args);
}

const NxHost kHost = {NX_GAME_INTERFACE, sizeof(NxHost), rt_host_svc, rt_host_jump, rt_host_break, rt_host_lookup,
                      Print};

#ifdef NXRT_GAME_MODULE
#ifdef _WIN32
constexpr const char* kModule = "nxgame.dll";
#else
constexpr const char* kModule = "libnxgame.so";
#endif

/// RT_GAME_MODULE=<file>, else `module` (the build folder's, a player start), else the module beside the executable.
NxGameEntry Entry(const char* module) {
    const char* named = std::getenv("RT_GAME_MODULE");
    const char* base = SDL_GetBasePath();
    const std::string path = named && *named ? named
                             : module         ? module
                                              : std::string(base ? base : "") + kModule;
    SDL_SharedObject* so = SDL_LoadObject(path.c_str());
    if (!so) Fatal("game: the game module %s did not load: %s (build it: nxrt --build, or Build on the start screen)", path.c_str(), SDL_GetError());
    const auto entry = reinterpret_cast<NxGameEntry>(SDL_LoadFunction(so, NX_GAME_ENTRY));
    if (!entry) Fatal("game: %s has no %s: not a game module", path.c_str(), NX_GAME_ENTRY);
    Log("game: module %s (loaded at t=%.2f s)", path.c_str(), Uptime());
    return entry; // (never unloaded)
}
#else
NxGameEntry Entry(const char*) {
    return nx_game;
}
#endif

} // namespace

void LoadGame(const char* module, const char* want_id) {
    const NxGame* g = Entry(module)(&kHost);
    if (!g || g->interface != NX_GAME_INTERFACE || g->size != sizeof(NxGame)) {
        Fatal("game: the game code was built for another runtime interface (this runtime: %d): rebuild it",
              NX_GAME_INTERFACE);
    }
    // (the builder checked nxgame.id beside the module; this is the module itself)
    if (want_id && std::strcmp(g->id, want_id) != 0) {
        Fatal("game: the game module was built from something else (its id %s, expected %s): build the game again",
              g->id, want_id);
    }
    g_game = g;
    Log("game: %lu functions, image 0x%llx bytes, %s memory%s, id %s", g->func_count,
        (unsigned long long)g->image_size, g->direct_base ? "direct" : "checked", g->mods ? ", mods built in" : "",
        g->id);
}

const NxGame& Game() {
    return *g_game;
}

bool GameLoaded() {
    return g_game != nullptr;
}

} // namespace rt
