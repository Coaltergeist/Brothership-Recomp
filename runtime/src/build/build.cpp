// The game builder (build.h).
#include "build.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

#include <SDL3/SDL.h>

#include "../mods.h"
#include "../options.h"
#include "../platform/platform.h"
#include "../rt.h"
#include "../seed/extract.h"
#include "flags.h"
#include "unpack.h"
#include "nxrt_toolchain_id.h" // NXRT_TOOLCHAIN_ID (runtime/cmake/toolchain_id.cmake)

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace rt::build {
namespace {

// One compile's peak memory, with room: 430 MB for the biggest lifted file (6.5 MB of C), ~300 MB typical
// (clang 23, -O1, 2026-11-01).
constexpr uint64_t kJobMemory = 450ull << 20;
// Estimates before this machine has timings of its own: bytes of C one job compiles per second with half the cores
// busy (2.42 GB in 310 s with 14 jobs on a development PC), the lift (12 s there), the link.
constexpr double kDefaultRate = 0.55e6, kDefaultLift = 15, kDefaultLink = 30;
constexpr double kDefaultSourceBytes = 2.45e9; // a whole lift's C, before there is one
constexpr double kDefaultExtract = 15, kDefaultSeed = 400; // seconds (10 s and 263-529 s on a development PC)

#ifdef _WIN32
constexpr const char* kExe = ".exe";
constexpr const char* kModule = "nxgame.dll";
constexpr const char* kTarget = "windows";
#else
constexpr const char* kExe = "";
constexpr const char* kModule = "libnxgame.so";
constexpr const char* kTarget = "linux";
#endif

std::string Env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

std::string Opt(const char* name) {
    const char* v = options::Get(name);
    return v ? v : "";
}

/// A path as SDL and the tools take it (UTF-8; forward slashes in response files suit clang and lld-link alike).
std::string Utf8(const fs::path& p) {
    const std::u8string s = p.u8string();
    return std::string(s.begin(), s.end());
}
std::string Slashed(const fs::path& p) {
    const std::u8string s = p.generic_u8string();
    return std::string(s.begin(), s.end());
}

bool Exists(const fs::path& p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

std::string Hex(const unsigned char* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) s += d[p[i] >> 4], s += d[p[i] & 15];
    return s;
}

/// FNV-1a, 64 bits: change detection, not security.
struct Hash {
    uint64_t h = 0xcbf29ce484222325ull;
    void Add(const void* p, size_t n) {
        for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const unsigned char*>(p)[i]) * 0x100000001b3ull;
    }
    void Add(const std::string& s) { Add(s.data(), s.size() + 1); } // (with the terminator: "a","bc" != "ab","c")
    std::string Text() const {
        unsigned char b[8];
        for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(h >> (56 - 8 * i));
        return Hex(b, 8);
    }
};

std::string ReadFile(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), {});
}

// --- The tools ----------------------------------------------------------------------------------------------------

struct Tools {
    fs::path root, lifter, clang, linker, game, kernel32, mods, hactool;
    std::string problem; // why they cannot be used
};

Tools FindTools() {
    Tools t;
    fs::path root;
    if (const std::string named = Env("RT_BUILD_TOOLS"); !named.empty()) {
        root = named;
    } else if (const char* base = SDL_GetBasePath()) {
        root = fs::path(base) / "tools";
    }
    if (Exists(root / (std::string("nxrecomp") + kExe))) {
        t.root = root;
        t.lifter = root / (std::string("nxrecomp") + kExe);
        t.clang = root / "llvm" / "bin" / (std::string("clang") + kExe);
        t.game = root / "game";
        t.mods = root / "mods";
        t.hactool = root / (std::string("hactool") + kExe);
    } else {
#if defined(NXRT_SOURCE_DIR) && !defined(_WIN32)
        // a Linux source checkout: the repository's own lifter build and compiler (README "Linux build")
        const fs::path src = NXRT_SOURCE_DIR;
        t.root = src;
        t.lifter = src / "build" / "lifter" / "nxrecomp";
        t.clang = src / "tools" / "llvm" / "bin" / "clang";
        t.game = src / "runtime" / "src" / "game";
        t.mods = src / "mods";
        t.hactool = src / "build" / "hactool" / "hactool"; // (tools/hactool-build)
#else
        t.problem = "The builder's tools are missing (no " + root.string() + "): reinstall Brothership Recomp.";
        return t;
#endif
    }
#ifdef _WIN32
    t.linker = t.clang.parent_path() / "lld-link.exe";
    t.kernel32 = t.game / "kernel32.lib";
#else
    t.linker = t.clang.parent_path() / "ld.lld";
#endif
    for (const fs::path& need : {t.lifter, t.clang, t.linker, t.game / "entry.c", t.game / "libc.c",
                                 t.game / "freestanding"}) {
        if (!Exists(need)) {
            t.problem = "The builder's tools are incomplete (" + need.string() + " is missing): reinstall.";
            return t;
        }
    }
#ifdef _WIN32
    if (!Exists(t.kernel32)) t.problem = "The builder's tools are incomplete (" + t.kernel32.string() + " is missing).";
#endif
    return t;
}

// --- What a build is made from ------------------------------------------------------------------------------------

struct Module {
    fs::path file;
    std::string build_id; // hex, 32 bytes
};

/// The game's code modules in load order (as the lifter links them): rtld, main, subsdk0-9, sdk - as dumped, or
/// hactool's uncompressed copies (<name>.bin) when only those are there.
std::vector<Module> GameModules(const fs::path& exefs) {
    std::vector<std::string> names{"rtld", "main"};
    for (int i = 0; i < 10; ++i) names.push_back("subsdk" + std::to_string(i));
    names.push_back("sdk");
    std::vector<Module> out;
    for (const std::string& n : names) {
        fs::path f = exefs / n;
        if (!Exists(f) && Exists(exefs / (n + ".bin"))) f = exefs / (n + ".bin");
        std::ifstream in(f, std::ios::binary);
        unsigned char h[0x60];
        if (!in || !in.read(reinterpret_cast<char*>(h), sizeof h) || std::memcmp(h, "NSO0", 4) != 0) continue;
        out.push_back({f, Hex(h + 0x40, 0x20)});
    }
    return out;
}

bool HasPatches(const fs::path& mod) {
    std::error_code ec;
    if (!fs::is_directory(mod / "exefs", ec)) return false;
    for (const auto& e : fs::recursive_directory_iterator(mod / "exefs", ec)) {
        const std::string ext = e.path().extension().string();
        if (e.is_regular_file() && (ext == ".pchtxt" || ext == ".ips")) return true;
    }
    return false;
}

/// The exefs mods lifted in (each becomes a switch, nxrecomp --mod): the ones that ship with the runtime (tools/mods:
/// the 60 fps patch) and the player's mods folder's, by name (the player's wins over a shipped one of the same name).
std::map<std::string, fs::path> LiftedMods(const Tools& t) {
    std::map<std::string, fs::path> out;
    std::error_code ec;
    for (const fs::path& dir : {t.mods, fs::path(mods::Folder())}) {
        for (const auto& e : fs::directory_iterator(dir, ec)) {
            if (e.is_directory() && HasPatches(e.path())) out[e.path().filename().string()] = e.path();
        }
    }
    return out;
}

struct Plan {
    Tools tools;
    Paths paths;
    std::vector<Module> modules;
    std::map<std::string, fs::path> mods;
    std::string id;
};

/// Everything a build needs and its id; false (why) when it cannot be built.
bool MakePlan(Plan& p, std::string* why) {
    p.paths = Where();
    if (p.paths.game.empty()) {
        if (why) *why = "Choose the game files first (Settings: Game files).";
        return false;
    }
    if (p.paths.packed && UnpackNeeded(p.paths.source, p.paths.game)) {
        if (why) *why = p.paths.keys.empty() ? "Choose your keys (Settings: Keys), then build: the game files are unpacked first."
                                           : "The game files are unpacked and built when you build.";
        return false;
    }
    const fs::path exefs = fs::path(p.paths.game) / "exefs";
    p.modules = GameModules(exefs);
    if (std::none_of(p.modules.begin(), p.modules.end(), [](const Module& m) { return m.file.stem() == "main"; })) {
        std::error_code ec;
        bool nsp = false;
        for (const auto& e : fs::directory_iterator(fs::path(p.paths.game), ec)) {
            if (e.path().extension() == ".nsp" || e.path().extension() == ".xci") nsp = true;
        }
        if (why) {
            *why = nsp ? "This folder holds packed game files (.nsp / .xci): choose the extracted dump, the folder with "
                         "exefs and romfs in it."
                       : "No game code in " + p.paths.game + ": choose the extracted dump, the folder with exefs and "
                         "romfs in it.";
        }
        return false;
    }
    p.tools = FindTools();
    if (!p.tools.problem.empty()) {
        if (why) *why = p.tools.problem;
        return false;
    }
    p.mods = LiftedMods(p.tools);
    Hash h;
    h.Add(kTarget);
    for (const std::string& f : LiftFlags()) h.Add(f);
    for (const Module& m : p.modules) h.Add(m.file.stem().string() + " " + m.build_id);
    for (const auto& [name, dir] : p.mods) {
        h.Add("mod " + name);
        std::vector<fs::path> files;
        std::error_code ec;
        for (const auto& e : fs::recursive_directory_iterator(dir / "exefs", ec)) {
            if (e.is_regular_file()) files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        for (const fs::path& f : files) h.Add(Slashed(fs::relative(f, dir, ec))), h.Add(ReadFile(f));
    }
    const Module& main = *std::find_if(p.modules.begin(), p.modules.end(),
                                       [](const Module& m) { return m.file.stem() == "main"; });
    p.id = std::string(NXRT_TOOLCHAIN_ID) + "-" + main.build_id.substr(0, 16) + "-" + h.Text();
    return true;
}

/// nxgame.id: `id=<id>` and a `mod=<name>` line per lifted mod.
std::map<std::string, std::vector<std::string>> ReadIdFile(const std::string& path) {
    std::map<std::string, std::vector<std::string>> out;
    std::ifstream f{fs::path(path)};
    for (std::string line; std::getline(f, line);) {
        if (const size_t eq = line.find('='); eq != std::string::npos) out[line.substr(0, eq)].push_back(line.substr(eq + 1));
    }
    return out;
}

// --- The running build --------------------------------------------------------------------------------------------

std::mutex g_mu;
Status g_status;           // (g_mu)
std::atomic<bool> g_running{false}, g_cancel{false};
std::ofstream g_log;       // the build folder's build.log (the build thread's)

void Note(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void Note(const char* fmt, ...) {
    char line[2048];
    va_list a;
    va_start(a, fmt);
    std::vsnprintf(line, sizeof line, fmt, a);
    va_end(a);
    Log("build: %s", line);
    if (g_log) g_log << line << '\n' << std::flush;
}

void Set(Stage stage, const std::string& step) {
    std::lock_guard lock(g_mu);
    g_status.stage = stage;
    g_status.step = step;
}

/// A tool's run: its output (stdout and stderr) in `log`, no window (platform::StartChild).
platform::Child* Spawn(const std::vector<std::string>& args, const fs::path& log, std::string* problem,
                       const std::vector<std::string>& env = {}) {
    std::string error;
    platform::Child* p = platform::StartChild(args, Utf8(log), &error, env);
    if (!p) {
        // LASRecomp #337: Windows' Control Flow Guard / antivirus stopping a freshly unpacked tool shows up here
        *problem = "Could not start " + fs::path(args[0]).filename().string() + " (" + error +
                   "). If antivirus or Windows Security (Exploit protection) blocked it, allow the folder " +
                   fs::path(args[0]).parent_path().string() + " and build again.";
    }
    return p;
}

std::string Lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

/// The last non-empty line of a log.
std::string LastLine(const std::string& text) {
    size_t end = text.find_last_not_of("\r\n");
    if (end == std::string::npos) return "";
    const size_t begin = text.rfind('\n', end);
    return text.substr(begin == std::string::npos ? 0 : begin + 1, end - (begin == std::string::npos ? 0 : begin + 1) + 1);
}

/// Why a tool that ended with `code` failed, for the player.
std::string Explain(const std::string& what, int code, const fs::path& log) {
    std::string text = ReadFile(log);
    if (text.size() > 8192) text = text.substr(text.size() - 8192);
    const std::string low = Lower(text);
    const uint32_t u = static_cast<uint32_t>(code);
    bool oom = low.find("out of memory") != std::string::npos || low.find("allocation failed") != std::string::npos ||
               low.find("bad_alloc") != std::string::npos || low.find("cannot allocate memory") != std::string::npos;
#ifdef _WIN32
    oom = oom || u == 0xC0000017 || u == 0xC000012D; // STATUS_NO_MEMORY, STATUS_COMMITMENT_LIMIT
#else
    oom = oom || code == -SIGKILL; // (Linux's out-of-memory killer ends a process with SIGKILL, silently)
#endif
    if (oom) {
        return what + " ran out of memory. Close other programs, or build with fewer jobs (RT_BUILD_JOBS=<n>).";
    }
    char hex[16];
    std::snprintf(hex, sizeof hex, "0x%08X", u);
#ifdef _WIN32
    if (u == 0xC0000409) {
        return what + " was stopped by a Windows security check (" + hex +
               "; Control Flow Guard / Exploit protection). Allow the builder's tools folder in Windows Security.";
    }
    if (u == 0xC0000022) return what + " was denied access (" + hex + "; antivirus?).";
    if (u >= 0xC0000000) return what + " crashed (" + hex + "): " + LastLine(text);
#else
    if (code < 0) return what + " was ended by signal " + std::to_string(-code) + ".";
#endif
    return what + " failed (exit " + std::to_string(code) + "): " + LastLine(text);
}

std::map<std::string, double> ReadTimings(const fs::path& path) {
    std::map<std::string, double> t;
    std::ifstream f(path);
    std::string name;
    double secs;
    while (f >> name >> secs) t[name] = secs;
    return t;
}

void WriteTimings(const fs::path& path, const std::map<std::string, double>& t) {
    std::ofstream f(path, std::ios::trunc);
    for (const auto& [name, secs] : t) f << name << ' ' << secs << '\n';
}

/// Waits for one process, ending it on Cancel; its exit code, or INT32_MIN when cancelled.
int WaitOne(platform::Child* p) {
    int code = 0;
    while (!platform::ChildDone(p, &code)) {
        if (g_cancel) {
            platform::KillChild(p);
            platform::FreeChild(p);
            return INT32_MIN;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    platform::FreeChild(p);
    return code;
}

struct Source {
    fs::path file;
    std::string stem;
    double bytes = 0, estimate = 0; // estimate: seconds (timings, else bytes / rate)
    bool known = false;             // estimate from this machine's timings
};

/// Build's shader steps (extract, seed: Steps::kShaders), for `Run`; `fail` and `cancelled` as there.
bool Shaders(const Plan& p, int jobs, std::map<std::string, double>& timings, Clock::time_point t0,
             const std::function<bool(const std::string&)>& fail, const std::function<bool()>& cancelled) {
    const auto since = [&](Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); };
    std::string why;
    std::error_code ec;
    const fs::path dir = p.paths.dir;
    const double extract_estimate = timings.count("extract") ? timings["extract"] : kDefaultExtract;
    const double seed_estimate = timings.count("seed") ? timings["seed"] : kDefaultSeed;

    // 4. The game's shader stages, from its archives (seed/extract.h; 1.1 GB, deleted after the seed)
    Set(Stage::kExtract, "Extracting shaders");
    const auto extract_start = Clock::now();
    const fs::path stages = dir / "seed";
    fs::remove_all(stages, ec);
    seed::ExtractStats xs;
    const auto extract_progress = [&](double f) {
        std::lock_guard lock(g_mu);
        g_status.elapsed = since(t0);
        g_status.remaining = std::max(0.0, extract_estimate * (1 - f)) + seed_estimate;
    };
    if (!seed::ExtractStages(p.paths.romfs, Utf8(stages), jobs, extract_progress, g_cancel, &xs, &why)) {
        return g_cancel ? cancelled() : fail("Extracting the game's shaders failed: " + why);
    }
    timings["extract"] = since(extract_start);
    Note("shaders: %zu stages from %zu chunks of %zu containers in %.0f s (%zu chunks unpacked differently from their "
         "hash: seed/extract.cpp)", xs.stages, xs.chunks, xs.containers, timings["extract"], xs.bad_chunks);

    // 5. The seed: this program run headless translates them into the pipeline cache and the shader store
    //    (RT_NVN_SEED_ONLY: the game boots to its first NVN call, seeds, ends; nothing if done before for this driver)
    Set(Stage::kSeed, "Building the shader cache");
    const auto seed_start = Clock::now();
    const fs::path seed_log = dir / "seed.log";
    platform::Child* proc = Spawn({platform::ExecutablePath()}, seed_log, &why,
                 {"RT_HEADLESS=1", "RT_NVN_SEED_ONLY=1", "RT_NVN_SEED_FRESH=1", "RT_NVN_SEED=" + Utf8(stages), "RT_GAME_DIR=" + p.paths.game,
                  "RT_BUILD_DIR=" + p.paths.dir});
    if (!proc) return fail(why);
    int seed_code = 0;
    for (int tick = 0; !platform::ChildDone(proc, &seed_code); ++tick) {
        if (g_cancel) {
            platform::KillChild(proc);
            platform::FreeChild(proc);
            return cancelled();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (tick % 10) continue;
        // its "nvn seed: <n> of <all> stages" lines
        std::string text = ReadFile(seed_log);
        const size_t at = text.rfind("nvn seed: ");
        unsigned long n = 0, all = 0;
        double secs = since(seed_start);
        if (at != std::string::npos) std::sscanf(text.c_str() + at, "nvn seed: %lu of %lu stages", &n, &all);
        std::lock_guard lock(g_mu);
        g_status.elapsed = since(t0);
        g_status.done = static_cast<int>(n), g_status.total = static_cast<int>(all);
        g_status.remaining = n > 0 && all > n ? secs * static_cast<double>(all - n) / static_cast<double>(n)
                                              : std::max(1.0, seed_estimate - secs);
    }
    platform::FreeChild(proc);
    if (seed_code != 0) return fail(Explain("Building the shader cache", seed_code, seed_log) + " (log: " + seed_log.string() + ")");
    timings["seed"] = since(seed_start);
    {
        const std::string text = ReadFile(seed_log);
        const size_t at = text.rfind("nvn seed: done");
        Note("seed: %.0f s: %s", timings["seed"],
             at == std::string::npos ? "?" : text.substr(at, text.find('\n', at) - at).c_str());
    }
    fs::remove_all(stages, ec);
    WriteTimings(dir / "timings.txt", timings);
    Note("done in %.0f s", since(t0));
    return true;
}

/// The build itself (the build thread): `steps` of it (build.h Steps). False (g_status.problem) on failure or cancel.
bool Run(int jobs, Steps steps) {
    const auto t0 = Clock::now();
    const auto since = [&](Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); };
    std::string why;
    std::error_code ec;
    const Paths where = Where();
    const fs::path dir = where.dir, gen = where.gen, obj = where.obj;
    fs::create_directories(obj, ec);
    fs::create_directories(gen, ec);
    if (ec) {
        std::lock_guard lock(g_mu);
        g_status.problem = "Could not make the build folder " + dir.string() + ": " + ec.message();
        return false;
    }
    g_log.open(fs::path(where.log), steps == Steps::kShaders ? std::ios::app : std::ios::trunc); // (after the game's)
    std::map<std::string, double> timings = ReadTimings(dir / "timings.txt");
    const auto fail = [&](const std::string& problem) {
        Note("failed after %.0f s: %s", since(t0), problem.c_str());
        WriteTimings(dir / "timings.txt", timings);
        std::lock_guard lock(g_mu);
        g_status.problem = problem;
        return false;
    };
    const auto cancelled = [&] {
        Note("cancelled after %.0f s", since(t0));
        WriteTimings(dir / "timings.txt", timings);
        return false;
    };

    if (steps == Steps::kShaders) { // (`nxrt --seed`: the module is built; its game files are unpacked)
        Plan p;
        if (!MakePlan(p, &why)) return fail(why);
        Note("shaders: game %s; folder %s", p.paths.game.c_str(), p.paths.dir.c_str());
        return Shaders(p, jobs, timings, t0, fail, cancelled);
    }

    // 0. The player's .nsp files unpacked, when the Game files folder holds those (unpack.h)
    if (where.packed && UnpackNeeded(where.source, where.game)) {
        const Tools t = FindTools();
        if (!t.problem.empty()) return fail(t.problem);
        if (!Exists(t.hactool)) return fail("The builder's tools are incomplete (" + t.hactool.string() + " is missing): reinstall.");
        Set(Stage::kUnpack, "Unpacking the game files");
        const auto unpack_start = Clock::now();
        Note("unpack: %s -> %s", where.source.c_str(), where.game.c_str());
        fs::remove(where.id_file, ec);
        const auto progress = [&](double f, const char* step) {
            std::lock_guard lock(g_mu);
            g_status.step = step;
            g_status.elapsed = since(t0);
            g_status.remaining = -1; // (the build's share is known once the game files are)
            if (f > 0.02) g_status.remaining = since(unpack_start) * (1 - f) / f;
        };
        if (!Unpack({where.source, where.keys, Utf8(t.hactool), where.game, Utf8(dir / "unpack"), Utf8(dir / "unpack.log")},
                    progress, g_cancel, &why)) {
            return g_cancel ? cancelled() : fail(why);
        }
        timings["unpack"] = since(unpack_start);
        Note("unpack: %.0f s", timings["unpack"]);
    }
    Plan p;
    if (!MakePlan(p, &why)) return fail(why);
    fs::remove(p.paths.id_file, ec); // not ready until this build has finished
    {
        const Module& main = *std::find_if(p.modules.begin(), p.modules.end(), [](const Module& m) { return m.file.stem() == "main"; });
        if (main.build_id.rfind("bf0607ac795b593a", 0) != 0) { // (1.0.1: the 60 fps patch's build id)
            Note("note: the game's main is build %s, not 1.0.1's (bf0607ac795b593a...): this beta is made for 1.0.1",
                 main.build_id.substr(0, 16).c_str());
        }
    }
    Note("id %s; game %s; folder %s; tools %s; %d jobs (%.1f GB free)", p.id.c_str(), p.paths.game.c_str(),
         p.paths.dir.c_str(), p.tools.root.string().c_str(), jobs, platform::FreeMemory() / 1e9);
    const double link_estimate = timings.count("link") ? timings["link"] : kDefaultLink;
    const double extract_estimate = timings.count("extract") ? timings["extract"] : kDefaultExtract;
    const double seed_estimate = timings.count("seed") ? timings["seed"] : kDefaultSeed;

    // 1. The lift. Unchanged files are left alone (the lifter's WriteFile), so their objects stay current.
    {
        std::vector<std::string> args{Utf8(p.tools.lifter)};
        for (const Module& m : p.modules) args.push_back(Utf8(m.file));
        args.push_back("-o");
        args.push_back(Utf8(gen));
        for (const std::string& f : LiftFlags()) args.push_back(f);
        for (const auto& [name, mod] : p.mods) args.push_back("--mod"), args.push_back(Utf8(mod));
        std::string mods;
        for (const auto& [name, mod] : p.mods) mods += (mods.empty() ? "" : ", ") + name;
        Note("lift: %zu modules, mods: %s", p.modules.size(), mods.empty() ? "none" : mods.c_str());
        const double lift_estimate = timings.count("lift") ? timings["lift"] : kDefaultLift;
        // the compile's share, when every object is to be made (a first build): known before the lift
        bool fresh = true;
        for (const auto& e : fs::directory_iterator(obj, ec)) {
            if (e.path().extension() == ".o") { fresh = false; break; }
        }
        double compile_estimate = -1;
        if (fresh) {
            double bytes = 0;
            for (const auto& e : fs::directory_iterator(gen, ec)) {
                if (e.path().extension() == ".c") bytes += static_cast<double>(e.file_size(ec));
            }
            compile_estimate = (bytes > 0 ? bytes : kDefaultSourceBytes) / kDefaultRate / jobs;
        }
        Set(Stage::kLift, "Translating the game's code");
        const auto lift_start = Clock::now();
        platform::Child* proc = Spawn(args, dir / "lift.log", &why);
        if (!proc) return fail(why);
        int code = 0;
        while (!platform::ChildDone(proc, &code)) {
            if (g_cancel) {
                platform::KillChild(proc);
                platform::FreeChild(proc);
                return cancelled();
            }
            {
                std::lock_guard lock(g_mu);
                g_status.elapsed = since(t0);
                g_status.remaining = compile_estimate < 0 ? -1
                                                          : std::max(1.0, lift_estimate - since(lift_start)) +
                                                                compile_estimate + link_estimate + extract_estimate +
                                                                seed_estimate;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        platform::FreeChild(proc);
        if (code != 0) return fail(Explain("The lifter", code, dir / "lift.log"));
        timings["lift"] = since(lift_start);
        Note("lift: %.1f s", timings["lift"]);
    }

    // 2. The compile: every source whose object is older than it (all of them after a flags change), biggest first.
    // The module's own sources copied into the build folder (a file only when it changed, so its object stays
    // current): the flags then name only paths in the build folder, and moving the program does not recompile all
    // (2026-11-01: a runtime moved from \\wsl.localhost to C: recompiled 573 files).
    const fs::path game = dir / "src" / "game";
    for (const auto& e : fs::recursive_directory_iterator(p.tools.game, ec)) {
        if (!e.is_regular_file()) continue;
        const fs::path to = game / fs::relative(e.path(), p.tools.game, ec);
        const std::string data = ReadFile(e.path());
        if (Exists(to) && ReadFile(to) == data) continue;
        fs::create_directories(to.parent_path(), ec);
        std::ofstream(to, std::ios::binary | std::ios::trunc) << data;
    }
    {
        std::string rsp;
        std::vector<std::string> flags = CommonFlags();
        for (const std::string& f : {std::string("-isystem"), Slashed(game / "freestanding"), std::string("-I"),
                                     Slashed(gen), std::string("-I"), Slashed(game)}) {
            flags.push_back(f);
        }
        for (const std::string& f : flags) rsp += "\"" + f + "\"\n";
        if (ReadFile(obj / "flags.rsp") != rsp) {
            Note("compile flags changed (or a first build): every source is compiled");
            for (const auto& e : fs::directory_iterator(obj, ec)) {
                if (e.path().extension() == ".o") fs::remove(e.path(), ec);
            }
            std::ofstream(obj / "flags.rsp", std::ios::binary | std::ios::trunc) << rsp;
        }
        std::ofstream(obj / "nx_game_id.h", std::ios::trunc) << "#define NX_GAME_ID_STRING \"" << p.id << "\"\n";
    }
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(gen, ec)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("funcs_", 0) == 0 && e.path().extension() == ".c") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    for (const char* n : {"nx_table.c", "recomp_runtime.c"}) files.push_back(gen / n);
    for (const char* n : {"entry.c", "libc.c"}) files.push_back(game / n);
    std::vector<Source> todo;
    std::vector<fs::path> objects;
    double rate = kDefaultRate;
    for (const fs::path& f : files) {
        Source s;
        s.file = f;
        s.stem = f.stem().string();
        const fs::path o = obj / (s.stem + ".o");
        objects.push_back(o);
        // entry.c always (it carries the id); the others when older than their source
        if (s.stem != "entry" && Exists(o) && fs::last_write_time(o, ec) >= fs::last_write_time(f, ec)) continue;
        s.bytes = static_cast<double>(fs::file_size(f, ec));
        s.known = timings.count(s.stem) > 0;
        s.estimate = s.known ? timings[s.stem] : s.bytes / rate;
        todo.push_back(s);
    }
    std::stable_sort(todo.begin(), todo.end(), [](const Source& a, const Source& b) { return a.estimate > b.estimate; });
    Note("compile: %zu of %zu sources", todo.size(), files.size());
    {
        std::lock_guard lock(g_mu);
        g_status.stage = Stage::kCompile;
        g_status.step = "Compiling";
        g_status.total = static_cast<int>(todo.size());
        g_status.done = 0;
    }
    struct Running {
        size_t index;
        platform::Child* proc;
        Clock::time_point start;
    };
    std::vector<Running> running;
    size_t next = 0;
    int done = 0;
    double done_bytes = 0, done_secs = 0; // of unknown-timing sources: this run's rate
    const auto kill_all = [&] {
        for (Running& r : running) {
            platform::KillChild(r.proc);
            platform::FreeChild(r.proc);
            fs::remove(obj / (todo[r.index].stem + ".o.tmp"), ec);
        }
        running.clear();
    };
    while (next < todo.size() || !running.empty()) {
        if (g_cancel) {
            kill_all();
            return cancelled();
        }
        while (next < todo.size() && static_cast<int>(running.size()) < jobs) {
            const Source& s = todo[next];
            std::vector<std::string> args{Utf8(p.tools.clang), "@" + Utf8(obj / "flags.rsp"),
                                          OptFlag(s.file.filename().string())};
            if (s.stem == "entry") args.push_back("-include"), args.push_back(Utf8(obj / "nx_game_id.h"));
            for (const std::string& a : {std::string("-c"), Utf8(s.file), std::string("-o"),
                                         Utf8(obj / (s.stem + ".o.tmp"))}) {
                args.push_back(a);
            }
            platform::Child* proc = Spawn(args, obj / (s.stem + ".log"), &why);
            if (!proc) {
                kill_all();
                return fail(why);
            }
            running.push_back({next++, proc, Clock::now()});
        }
        for (size_t i = 0; i < running.size();) {
            int code = 0;
            if (!platform::ChildDone(running[i].proc, &code)) {
                ++i;
                continue;
            }
            platform::FreeChild(running[i].proc);
            const Source& s = todo[running[i].index];
            const double secs = since(running[i].start);
            running.erase(running.begin() + static_cast<std::ptrdiff_t>(i));
            const fs::path log = obj / (s.stem + ".log");
            if (code != 0) {
                kill_all();
                return fail(Explain("Compiling " + s.file.filename().string(), code, log) + " (log: " + log.string() + ")");
            }
            fs::rename(obj / (s.stem + ".o.tmp"), obj / (s.stem + ".o"), ec);
            if (ec) {
                kill_all();
                return fail("Could not write " + (obj / (s.stem + ".o")).string() + ": " + ec.message());
            }
            if (fs::file_size(log, ec) == 0) fs::remove(log, ec);
            timings[s.stem] = secs;
            if (!s.known) done_bytes += s.bytes, done_secs += secs;
            ++done;
        }
        // the estimate: what is left, spread over the jobs; unknown sources at this run's rate once there is one
        if (done_secs > 0 && done >= jobs) rate = done_bytes / done_secs;
        double left = 0;
        for (size_t k = next; k < todo.size(); ++k) left += todo[k].known ? todo[k].estimate : todo[k].bytes / rate;
        for (const Running& r : running) {
            const Source& s = todo[r.index];
            left += std::max(0.5, (s.known ? s.estimate : s.bytes / rate) - since(r.start));
        }
        {
            std::lock_guard lock(g_mu);
            g_status.done = done;
            g_status.elapsed = since(t0);
            g_status.remaining = left / jobs + link_estimate + extract_estimate + seed_estimate;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    Note("compile: %d sources in %.0f s", done, since(t0) - timings["lift"]);

    // 3. The link (the objects of this lift only: a file the lifter no longer writes is not linked).
    Set(Stage::kLink, "Linking");
    const auto link_start = Clock::now();
    const fs::path module = p.paths.module, tmp = module.string() + ".tmp";
    {
        std::string rsp;
        for (const std::string& f : LinkFlags(Slashed(tmp))) rsp += "\"" + f + "\"\n";
        for (const fs::path& o : objects) rsp += "\"" + Slashed(o) + "\"\n";
#ifdef _WIN32
        rsp += "\"" + Slashed(p.tools.kernel32) + "\"\n";
#endif
        std::ofstream(obj / "link.rsp", std::ios::binary | std::ios::trunc) << rsp;
    }
#ifdef _WIN32
    const fs::path linker = p.tools.linker;
#else
    const fs::path linker = p.tools.clang; // (clang drives ld.lld: -fuse-ld=lld)
#endif
    platform::Child* proc = Spawn({Utf8(linker), "@" + Utf8(obj / "link.rsp")}, dir / "link.log", &why);
    if (!proc) return fail(why);
    const int code = WaitOne(proc);
    if (code == INT32_MIN) return cancelled();
    if (code != 0) return fail(Explain("The link", code, dir / "link.log"));
    fs::rename(tmp, module, ec);
    if (ec) return fail("Could not replace " + module.string() + ": " + ec.message() + " (is the game running?)");
    timings["link"] = since(link_start);
    {
        std::ofstream f{fs::path(p.paths.id_file), std::ios::trunc};
        f << "id=" << p.id << '\n';
        for (const auto& [name, mod] : p.mods) f << "mod=" << name << '\n';
    }
    WriteTimings(dir / "timings.txt", timings);
    Note("built in %.0f s (lift %.0f, link %.0f): %s, %.0f MB", since(t0), timings["lift"], timings["link"],
         module.string().c_str(), static_cast<double>(fs::file_size(module, ec)) / 1e6);
    // (the game plays from here on: a failed or cancelled seed only means shaders translated at first use)
    if (steps == Steps::kGame) { // the start screen: it hands the shader steps to `nxrt --seed` (build.h Steps)
        Set(Stage::kSeedNext, "Building the shader cache next");
        Note("the shader cache next (nxrt --seed)");
        return true;
    }
    return Shaders(p, jobs, timings, t0, fail, cancelled);
}

int DefaultJobs() {
    if (const std::string e = Env("RT_BUILD_JOBS"); !e.empty()) return std::max(1, std::atoi(e.c_str()));
    int jobs = std::max(1, (SDL_GetNumLogicalCPUCores() + 1) / 2); // half the cores: the heat rule
    if (const uint64_t free = platform::FreeMemory()) {
        jobs = std::min<int>(jobs, static_cast<int>(std::max<uint64_t>(1, free / kJobMemory)));
    }
    return jobs;
}

} // namespace

std::string DefaultDir() {
#ifdef _WIN32
    const std::string base = Env("LOCALAPPDATA");
    return (fs::path(base.empty() ? "." : base) / "brothership-recomp" / "build").string();
#else
    if (const std::string x = Env("XDG_DATA_HOME"); !x.empty()) return (fs::path(x) / "brothership-recomp" / "build").string();
    const std::string home = Env("HOME");
    return (fs::path(home.empty() ? "." : home) / ".local" / "share" / "brothership-recomp" / "build").string();
#endif
}

Paths Where() {
    Paths p;
    p.source = Opt("RT_GAME_DIR");
    p.keys = Opt("RT_KEYS");
    p.dir = Opt("RT_BUILD_DIR");
    if (p.dir.empty()) p.dir = DefaultDir();
    p.packed = IsPacked(p.source); // (.nsp files: unpacked into the build folder's game/)
    p.game = p.packed ? (fs::path(p.dir) / "game").string() : p.source;
    const fs::path dir = p.dir, game = p.game;
    p.gen = (dir / "gen").string();
    p.obj = (dir / "obj").string();
    p.module = (dir / kModule).string();
    p.id_file = (dir / "nxgame.id").string();
    p.log = (dir / "build.log").string();
    p.image = (dir / "gen" / "image.bin").string();
    p.symbols = (dir / "gen" / "symbols.txt").string();
    if (!p.game.empty()) {
        p.romfs = (game / "romfs").string();
        p.npdm = (game / "exefs" / "main.npdm").string();
    }
    return p;
}

std::string WantedId(std::string* why) {
    Plan p;
    return MakePlan(p, why) ? p.id : "";
}

bool Ready(std::string* why) {
    Plan p;
    if (!MakePlan(p, why)) return false;
    for (const std::string& f : {p.paths.module, p.paths.image, p.paths.symbols, p.paths.id_file}) {
        if (!Exists(f)) {
            if (why) *why = "The game is not built yet.";
            return false;
        }
    }
    const auto ids = ReadIdFile(p.paths.id_file)["id"];
    if (ids.empty() || ids[0] != p.id) {
        if (why) *why = "The game needs a new build (the game files, the mods or this runtime changed).";
        return false;
    }
    return true;
}

std::vector<std::string> BuiltMods() {
    return ReadIdFile(Where().id_file)["mod"];
}

bool Start(Steps steps) {
    bool expected = false;
    if (!g_running.compare_exchange_strong(expected, true)) return false;
    g_cancel = false;
    const int jobs = DefaultJobs();
    {
        std::lock_guard lock(g_mu);
        g_status = Status{};
        g_status.stage = Stage::kLift;
        g_status.step = "Starting";
        g_status.jobs = jobs;
    }
    std::thread([jobs, steps] {
        const auto t0 = Clock::now();
        const bool ok = Run(jobs, steps);
        if (g_log.is_open()) g_log.close();
        {
            std::lock_guard lock(g_mu);
            g_status.elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
            g_status.remaining = 0;
            const Stage done = steps == Steps::kGame ? Stage::kSeedNext : Stage::kDone;
            g_status.stage = ok ? done : g_cancel ? Stage::kCancelled : Stage::kFailed;
            g_status.step = ok ? (done == Stage::kSeedNext ? "Building the shader cache next" : "Done")
                               : g_cancel ? "Cancelled" : "Failed";
        }
        g_running = false;
    }).detach();
    return true;
}

Status Get() {
    std::lock_guard lock(g_mu);
    return g_status;
}

void Cancel() {
    if (g_running) g_cancel = true;
}

bool SeedMarked() {
    const std::string named = Env("RT_PIPELINE_CACHE");
    const fs::path cache = !named.empty() && named != "0" ? fs::path(named) : fs::path(options::UserDir()) / "cache";
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(cache, ec)) {
        if (e.path().filename().string().ends_with(".nvn.seed")) return true;
    }
    return false;
}

bool Running() {
    return g_running;
}

void Stop() {
    Cancel();
    for (int i = 0; i < 200 && g_running; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

const char* StageName(Stage s) {
    switch (s) {
    case Stage::kIdle: return "idle";
    case Stage::kUnpack: return "unpack";
    case Stage::kLift: return "lift";
    case Stage::kCompile: return "compile";
    case Stage::kLink: return "link";
    case Stage::kExtract: return "shaders";
    case Stage::kSeed: return "seed";
    case Stage::kSeedNext: return "seed next";
    case Stage::kDone: return "done";
    case Stage::kFailed: return "failed";
    case Stage::kCancelled: return "cancelled";
    }
    return "?";
}

namespace {
std::string MinSec(double secs) {
    const int s = static_cast<int>(secs + 0.5);
    char b[32];
    std::snprintf(b, sizeof b, "%d:%02d", s / 60, s % 60);
    return b;
}
} // namespace

int Command() {
    std::signal(SIGINT, [](int) { g_cancel = true; }); // Ctrl+C: Cancel (only the flag: safe in a handler)
    std::string why;
    const Paths p = Where();
    const bool ready = Ready(&why);
    if (ready && SeedMarked()) {
        std::printf("The game is built and up to date: %s\n", p.module.c_str());
        return 0;
    }
    if (ready) std::printf("Building the shader cache (the game is built)\n  build folder: %s\n", p.dir.c_str());
    else std::printf("Building the game: %s\n  game files: %s\n  build folder: %s\n", why.c_str(),
                     p.game.empty() ? "(not chosen)" : p.game.c_str(), p.dir.c_str());
    std::fflush(stdout);
    Start(ready ? Steps::kShaders : Steps::kAll);
    auto last = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    Stage shown = Stage::kIdle;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const Status s = Get();
        const bool end = s.stage == Stage::kDone || s.stage == Stage::kFailed || s.stage == Stage::kCancelled;
        if (end && g_running) continue; // (the thread's last words)
        const auto now = std::chrono::steady_clock::now();
        if (s.stage != shown || now - last >= std::chrono::seconds(5) || end) {
            std::string line = "[" + std::string(StageName(s.stage)) + "] " + MinSec(s.elapsed);
            if (s.stage == Stage::kCompile) {
                line += "  " + std::to_string(s.done) + "/" + std::to_string(s.total) + " files, " +
                        std::to_string(s.jobs) + " jobs";
            }
            if (s.stage == Stage::kSeed && s.total) line += "  " + std::to_string(s.done) + "/" + std::to_string(s.total) + " shaders";
            if (!end && s.remaining >= 0) line += "  about " + MinSec(s.remaining) + " left";
            std::printf("%s\n", line.c_str());
            std::fflush(stdout);
            shown = s.stage, last = now;
        }
        if (!end) continue;
        if (s.stage == Stage::kDone) std::printf("Built: %s\n", p.module.c_str());
        if (s.stage == Stage::kFailed) std::printf("Build failed: %s\nThe log: %s\n", s.problem.c_str(), p.log.c_str());
        if (s.stage == Stage::kCancelled) std::printf("Cancelled (the files compiled so far are kept).\n");
        std::fflush(stdout);
        return s.stage == Stage::kDone ? 0 : s.stage == Stage::kCancelled ? 130 : 1;
    }
}

} // namespace rt::build
