// The player's options (options.h).
#include "options.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>

#include "rt.h"

namespace rt::options {
namespace {

struct Settings {
    std::mutex mu;
    bool loaded = false;
    std::map<std::string, std::string> values; // the file's, by option name
    std::set<std::string> told;                 // file values ignored (logged once)
    std::set<std::string> kept;                 // every value Get returned from the file: its pointers stay valid
};

Settings& S() {
    static Settings* s = new Settings; // used until the process ends
    return *s;
}

const Option& Find(const char* name) {
    for (const Option& o : All()) {
        if (std::strcmp(o.name, name) == 0) return o;
    }
    Fatal("options: %s is not a player option (options.cpp All)", name);
}

bool Allowed(const Option& o, const std::string& v) {
    if (std::strchr(o.values, '<')) return true; // free text
    return (std::string("|") + o.values + "|").find("|" + v + "|") != std::string::npos;
}

std::string Trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

/// Test runs (headless, or scripted input: the bench scripts, M1, difftest) never read the player's settings: their
/// results must not depend on what someone chose on the start screen (2026-10-31: M1 on Windows took the user's
/// RT_MODS and stopped). RT_SETTINGS=<file> is read even then (the start screen's own tests).
bool TestRun() {
    const char* headless = std::getenv("RT_HEADLESS");
    const char* input = std::getenv("RT_INPUT");
    return (headless && *headless && std::strcmp(headless, "0") != 0) || (input && *input);
}

/// Reads the settings file once (with S().mu held). Unknown names are dropped (an older or newer build's).
void Load(Settings& s) {
    if (s.loaded) return;
    s.loaded = true;
    if (TestRun() && !std::getenv("RT_SETTINGS")) return;
    std::ifstream f{std::filesystem::path(SettingsPath())};
    for (std::string line; std::getline(f, line);) {
        line = Trim(line);
        const size_t eq = line.find('=');
        if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
        const std::string name = Trim(line.substr(0, eq)), value = Trim(line.substr(eq + 1));
        for (const Option& o : All()) {
            if (name == o.name && !value.empty()) s.values[name] = value;
        }
    }
}

const char* Env(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? v : nullptr;
}

} // namespace

const std::vector<Option>& All() {
    static const std::vector<Option> all{
        {"RT_RES", "Resolution", "auto|360p|540p|720p|1080p|1440p|2160p|4k",
         "Auto|360p|540p|720p|1080p|1440p|2160p (4K)|", "720p"}, // (auto: the window's height, 720p-1440p)
        {"RT_AA", "Anti-aliasing", "off|smaa", "Off|SMAA", "off"},
        {"RT_WINDOW", "Window mode", "windowed|borderless", "Windowed|Borderless", "windowed"},
        {"RT_WINDOW_SIZE", "Window size", "<width>x<height>", nullptr, "1280x720"},
        {"RT_FPS", "Frame rate", "30|60|uncapped", "30|60|Uncapped", "30"}, // (60 and up need the 60 fps mod)
        {"RT_VSYNC", "VSync", "on|off", "On|Off", "on"},
        {"RT_BUTTONS", "Button layout", "positions|labels", "Switch positions|Controller labels", "positions"},
        {"RT_LANGUAGE", "Language", "en-US|en-GB|fr|fr-CA|de|it|es|es-419|nl|pt-BR|ja|ko|zh-Hans|zh-Hant",
         "English (US)|English (UK)|French|French (Canada)|German|Italian|Spanish|Spanish (Latin America)|Dutch|"
         "Portuguese (Brazil)|Japanese|Korean|Chinese (Simplified)|Chinese (Traditional)",
         "en-US"}, // (the game's 14, by its per-language assets; am's GetDesiredLanguage answers it)
        {"RT_DEBUG", "Debug mode", "off|on", "Off|On", "off"}, // (the FPS box, F7-F10: Debug())
        {"RT_MODS", "Mods", "<mod>[,<mod>...]|all|none", nullptr, "none"},
        {"RT_GAME_DIR", "Game files", "<folder>", nullptr, ""},   // (the dump: exefs/ + romfs/, build/build.h)
        {"RT_KEYS", "Keys", "<folder>", nullptr, ""},             // (the folder with prod.keys: unpacking .nsp files)
        {"RT_BUILD_DIR", "Build folder", "<folder>", nullptr, ""}, // ("": build::DefaultDir())
    };
    return all;
}

const char* Get(const char* name) {
    const Option& o = Find(name);
    if (const char* v = Env(name)) {
        if (!Allowed(o, v)) Fatal("%s=%s: expected %s", name, v, o.values);
        return v;
    }
    Settings& s = S();
    std::lock_guard lock(s.mu);
    Load(s);
    const auto it = s.values.find(name);
    if (it == s.values.end()) return nullptr;
    if (!Allowed(o, it->second)) {
        if (s.told.insert(name).second) {
            Log("options: %s=%s in %s is not one of %s: the default instead", name, it->second.c_str(),
                SettingsPath().c_str(), o.values);
        }
        return nullptr;
    }
    return s.kept.insert(it->second).first->c_str();
}

const char* Source(const char* name) {
    Find(name);
    if (Env(name)) return "environment";
    return Get(name) ? "settings" : "default";
}

std::string Saved(const char* name) {
    Find(name);
    Settings& s = S();
    std::lock_guard lock(s.mu);
    Load(s);
    const auto it = s.values.find(name);
    return it == s.values.end() ? "" : it->second;
}

void Save(const char* name, const std::string& value) {
    Find(name);
    Settings& s = S();
    std::lock_guard lock(s.mu);
    Load(s);
    if (value.empty()) s.values.erase(name);
    else s.values[name] = value;
}

bool Write() {
    Settings& s = S();
    std::lock_guard lock(s.mu);
    Load(s);
    const std::filesystem::path path = std::filesystem::path(SettingsPath());
    std::error_code ec;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream f(path, std::ios::trunc);
    f << "# Brothership Recomp settings, written by the start screen. One option per line, NAME=value;\n"
         "# a missing line means the default. An environment variable of the same name wins over this file.\n";
    for (const Option& o : All()) {
        const auto it = s.values.find(o.name);
        if (it != s.values.end()) f << o.name << '=' << it->second << '\n';
    }
    f.close();
    if (!f) {
        Log("options: could not write %s", path.string().c_str());
        return false;
    }
    Log("options: saved %s", path.string().c_str());
    return true;
}

bool Debug() {
    const char* v = Get("RT_DEBUG");
    return v && std::strcmp(v, "on") == 0;
}

std::string UserDir() {
#ifdef _WIN32
    const char* base = Env("LOCALAPPDATA");
    return base ? (std::filesystem::path(base) / "brothership-recomp").string() : "brothership-recomp";
#else
    if (const char* x = Env("XDG_CONFIG_HOME")) return (std::filesystem::path(x) / "brothership-recomp").string();
    const char* home = Env("HOME");
    return home ? (std::filesystem::path(home) / ".config" / "brothership-recomp").string() : "brothership-recomp";
#endif
}

std::string SettingsPath() {
    if (const char* f = Env("RT_SETTINGS")) return f;
    return (std::filesystem::path(UserDir()) / "settings.ini").string();
}

void LogAll() {
    std::string line;
    for (const Option& o : All()) {
        const char* v = Get(o.name);
        const char* from = Source(o.name);
        line += std::string(line.empty() ? "" : ", ") + o.name + "=" + (v ? v : std::string("(") + o.fallback + ")");
        if (std::strcmp(from, "settings") == 0) line += " [settings]";
    }
    Log("options: %s (settings file %s%s)", line.c_str(), SettingsPath().c_str(),
        TestRun() && !std::getenv("RT_SETTINGS") ? ": not read, a test run" : "");
}

} // namespace rt::options
