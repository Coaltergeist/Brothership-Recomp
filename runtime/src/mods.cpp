// The player's mods (mods.h).
#include "mods.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "options.h"
#include "rt.h"

namespace rt::mods {
namespace {

namespace fs = std::filesystem;

std::vector<std::string> g_enabled;
bool g_done = false;

fs::path Dir() {
    const char* d = std::getenv("RT_MODS_DIR");
    return d && *d ? fs::path(d) : fs::path(options::UserDir()) / "mods";
}

// A folder's exefs/ holds patches (.pchtxt / .ips) for some game version.
bool HasExefs(const fs::path& mod) {
    std::error_code ec;
    if (!fs::is_directory(mod / "exefs", ec)) return false;
    for (const auto& e : fs::recursive_directory_iterator(mod / "exefs", ec)) {
        const std::string ext = e.path().extension().string();
        if (e.is_regular_file() && (ext == ".pchtxt" || ext == ".ips")) return true;
    }
    return false;
}

} // namespace

const std::vector<std::string>& Enabled(const std::vector<std::string>& built) {
    if (g_done) return g_enabled;
    g_done = true;
    std::vector<std::string> want;
    const char* v = options::Get("RT_MODS");
    for (std::string s = v ? v : ""; !s.empty();) {
        const size_t comma = s.find(',');
        if (std::string name = s.substr(0, comma); !name.empty() && name != "none") want.push_back(name);
        s = comma == std::string::npos ? "" : s.substr(comma + 1);
    }
    // the frame-rate patch follows the Frame rate setting (kFrameRateMod)
    const char* fps = options::Get("RT_FPS");
    const bool faster = fps && (std::strcmp(fps, "60") == 0 || std::strcmp(fps, "uncapped") == 0);
    if (faster && std::find(want.begin(), want.end(), kFrameRateMod) == want.end()) {
        if (std::find(built.begin(), built.end(), kFrameRateMod) != built.end()) {
            want.push_back(kFrameRateMod);
            Log("mods: %s on for RT_FPS=%s", kFrameRateMod, fps);
        } else {
            Log("mods: RT_FPS=%s, but this build has no frame-rate patch (%s, nxrecomp --mod): the game stays at 30 fps",
                fps, kFrameRateMod);
        }
    }
    if (want.empty()) return g_enabled;
    const fs::path dir = Dir();
    const std::vector<std::string> folders = Folders(); // the mods folder's mods, by name
    const auto in = [](const std::vector<std::string>& list, const std::string& name) {
        return std::find(list.begin(), list.end(), name) != list.end();
    };
    const bool all = in(want, "all");
    if (all) {
        want = built;
        for (const auto& f : folders) {
            if (!in(want, f)) want.push_back(f);
        }
        std::sort(want.begin(), want.end());
    }
    std::string have; // for the messages
    for (const auto& n : built) have += (have.empty() ? "" : ", ") + n + " (in the build)";
    for (const auto& f : folders) {
        if (!in(built, f)) have += (have.empty() ? "" : ", ") + f;
    }
    for (const std::string& name : want) {
        const bool folder = in(folders, name);
        if (!in(built, name) && !folder) {
            Fatal("RT_MODS: no mod named %s (mods: %s; the folder: RT_MODS_DIR=%s)", name.c_str(),
                  have.empty() ? "none" : have.c_str(), dir.string().c_str());
        }
        // A patch the build lacks would silently do nothing: say so (a mod for another game version only matches
        // no module at lift time, and `nxrecomp --mod` refuses that too).
        if (folder && !in(built, name) && HasExefs(dir / name)) {
            if (all) {
                Log("mods: %s skipped: its exefs patches are not in this build", name.c_str());
                continue;
            }
            Fatal("RT_MODS: mod %s has exefs patches, but this build was lifted without them: re-lift with "
                  "nxrecomp --mod \"<mods folder>/%s\"", name.c_str(), name.c_str());
        }
        g_enabled.push_back(name);
    }
    return g_enabled;
}

std::string Folder() {
    return Dir().string();
}

std::vector<std::string> Folders() {
    std::vector<std::string> folders;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(Dir(), ec)) {
        if (e.is_directory()) folders.push_back(e.path().filename().string());
    }
    std::sort(folders.begin(), folders.end());
    return folders;
}

bool HasRomfs(const std::string& name) {
    std::error_code ec;
    return fs::is_directory(Dir() / name / "romfs", ec);
}

bool HasExefs(const std::string& name) {
    return HasExefs(Dir() / name);
}

std::vector<std::string> RomfsLayers() {
    std::vector<std::string> layers;
    const fs::path dir = Dir();
    for (const std::string& name : g_enabled) {
        std::error_code ec;
        if (fs::is_directory(dir / name / "romfs", ec)) layers.push_back((dir / name / "romfs").string());
    }
    return layers;
}

} // namespace rt::mods
