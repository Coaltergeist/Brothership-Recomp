// The start screen (start.h).
#include "start.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <SDL3/SDL.h>

#include "../nvn/display.h"
#include "../nvn/layer.h"
#include "../nvn/vk.h"

#include "imgui.h"
#include "imgui_internal.h" // (RegisterUserTexture: the background picture)
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"

#include "../build/build.h"
#include "../build/unpack.h"
#include "../input.h"
#include "../mods.h"
#include "../options.h"
#include "../platform/platform.h"
#include "../rt.h"
#include "../window.h"
#include "look.h"
#include "seed.h"

namespace rt::ui {
namespace {

enum class Page { kMain, kSettings, kMods, kBuild };

struct Screen {
    std::vector<std::string> built; // mods lifted into this build
    bool player = false;            // a player start: the game is built here (build.h), loaded at Play
    bool ready = true;              // (player) the build folder's module is current
    std::string not_ready;          // (player) why not
    std::chrono::steady_clock::time_point checked; // when `ready` was last worked out
    ImTextureData* background = nullptr; // the picture behind the pages, if there is one
    Page page = Page::kMain;
    bool play = false;
    char size[32] = {};       // the window size being edited
    std::string size_problem; // why the typed size was not taken
    std::map<std::string, std::string> folder_problem; // a typed folder's problem, by option
    std::map<std::string, std::array<char, 1024>> folder; // the folders being edited, by option
    std::string built_for;    // the game and build folders of the last Build press: its failure shows only for them
    bool focus = true;        // put the gamepad / keyboard focus on the page's first item
    bool handed_over = false; // the shader steps were handed to `--seed` (or that failed: handover_problem)
    std::string handover_problem;
};

/// Build's game steps are done (Stage::kSeedNext): the shader steps go to a run of this program of its own
/// (`--seed`, seed.h), and this one ends without writing the pipeline cache - the seed never runs beside a start
/// screen (build.h Steps).
void HandOver(Screen& s) {
    s.handed_over = true;
    if (!StartSelf("--seed", &s.handover_problem)) {
        s.handover_problem = "Could not start the shader cache's build: " + s.handover_problem;
        Log("start screen: %s", s.handover_problem.c_str());
        return;
    }
    Log("start screen: the shader cache is built by a run of its own (--seed): ending");
    nvn::KeepPipelineCacheFile();
    Exit(0);
}

/// A folder chosen in the system's dialog (SDL calls back on a thread of its own): applied by the next frame.
std::mutex g_picked_mu;
std::string g_picked_option, g_picked_folder;

/// (player) Works out whether the game is built, at most once a second (it reads the game's headers and mods).
void CheckReady(Screen& s, bool now = false) {
    if (!s.player) return;
    const auto t = std::chrono::steady_clock::now();
    if (!now && t - s.checked < std::chrono::seconds(1)) return;
    s.checked = t;
    if (build::Running()) return;
    s.not_ready.clear();
    s.ready = build::Ready(&s.not_ready);
    s.built = build::BuiltMods();
}

std::vector<std::string> Split(const std::string& s, char by) {
    std::vector<std::string> out;
    size_t at = 0;
    for (;;) {
        const size_t next = s.find(by, at);
        if (std::string part = s.substr(at, next - at); !part.empty()) out.push_back(part);
        if (next == std::string::npos) return out;
        at = next + 1;
    }
}

bool FromEnvironment(const options::Option& o) {
    return std::strcmp(options::Source(o.name), "environment") == 0;
}

void Saved(const char* name, const std::string& value) {
    options::Save(name, value);
    options::Write();
}

/// A setting with a fixed list of values: a choice among their names (an unset setting shows its default).
void ChoiceRow(const options::Option& o) {
    const std::vector<std::string> values = Split(o.values, '|');
    std::vector<std::string> names = values;
    if (o.names) { // (not Split: it drops empty parts, and an alias's "" says it is not offered)
        names.clear();
        const std::string all = std::string(o.names) + "|";
        for (size_t at = 0, bar; (bar = all.find('|', at)) != std::string::npos; at = bar + 1) {
            names.push_back(all.substr(at, bar - at));
        }
    }
    const std::string saved = options::Saved(o.name);
    const std::string current = saved.empty() ? o.fallback : saved;
    std::string shown = current;
    for (size_t i = 0; i < values.size() && i < names.size(); ++i) {
        if (values[i] == current && !names[i].empty()) shown = names[i];
    }
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo((std::string("##") + o.name).c_str(), shown.c_str())) {
        for (size_t i = 0; i < values.size() && i < names.size(); ++i) {
            if (names[i].empty()) continue;
            if (ImGui::Selectable(names[i].c_str(), values[i] == current)) Saved(o.name, values[i]);
        }
        ImGui::EndCombo();
    }
}

/// RT_WINDOW_SIZE: typed as <width>x<height>, taken when the field is left.
void SizeRow(Screen& s, const options::Option& o) {
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##size", o.fallback, s.size, sizeof s.size);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        int w = 0, h = 0;
        char rest = 0;
        const std::string typed = s.size;
        if (typed.empty()) {
            Saved(o.name, ""), s.size_problem.clear();
        } else if (std::sscanf(typed.c_str(), "%dx%d%c", &w, &h, &rest) == 2 && w >= 320 && h >= 180 && w <= 16384 &&
                   h <= 16384) {
            Saved(o.name, std::to_string(w) + "x" + std::to_string(h)), s.size_problem.clear();
        } else {
            s.size_problem = "\"" + typed + "\" is not a size like 1920x1080: not saved";
        }
    }
    if (!s.size_problem.empty()) ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), "%s", s.size_problem.c_str());
}

/// A folder setting (RT_GAME_DIR, RT_KEYS, RT_BUILD_DIR): typed, taken when the field is left, or chosen with Browse.
void FolderRow(Screen& s, const options::Option& o) {
    auto& buf = s.folder[o.name]; // (filled from the settings when the screen opens: Run)
    const bool game = std::strcmp(o.name, "RT_GAME_DIR") == 0, keys = std::strcmp(o.name, "RT_KEYS") == 0;
    const std::string hint = game ? "the folder with your .nsp files" : keys ? "the folder with prod.keys" : build::DefaultDir();
    const float browse = ImGui::CalcTextSize("Browse").x + ImGui::GetStyle().FramePadding.x * 2;
    ImGui::SetNextItemWidth(-browse - ImGui::GetStyle().ItemSpacing.x);
    ImGui::InputTextWithHint("##folder", hint.c_str(), buf.data(), buf.size());
    bool take = ImGui::IsItemDeactivatedAfterEdit();
    ImGui::SameLine();
    if (ImGui::Button("Browse")) {
        static std::string option; // (the callback's user data)
        option = o.name;
        SDL_ShowOpenFolderDialog(
            [](void* name, const char* const* list, int) {
                if (!list || !list[0]) return; // cancelled, or an error
                std::lock_guard lock(g_picked_mu);
                g_picked_option = static_cast<const std::string*>(name)->c_str();
                g_picked_folder = list[0];
            },
            &option, window::Sdl(), buf[0] ? buf.data() : nullptr, false);
    }
    {
        std::lock_guard lock(g_picked_mu);
        if (g_picked_option == o.name) {
            std::snprintf(buf.data(), buf.size(), "%s", g_picked_folder.c_str());
            g_picked_option.clear();
            take = true;
        }
    }
    if (take) {
        const std::string typed = buf.data();
        std::string& problem = s.folder_problem[o.name];
        problem.clear();
        std::error_code ec;
        const std::filesystem::path exefs = std::filesystem::path(typed) / "exefs";
        if (game && !typed.empty() && !build::IsPacked(typed) && !std::filesystem::exists(exefs / "main", ec) &&
            !std::filesystem::exists(exefs / "main.bin", ec)) {
            problem = "No .nsp files here";
        }
        if (keys && !typed.empty() && !std::filesystem::exists(std::filesystem::path(typed) / "prod.keys", ec) &&
            std::filesystem::path(typed).filename() != "prod.keys") {
            problem = "No prod.keys here";
        }
        Saved(o.name, typed);
        CheckReady(s, true);
    }
    if (!s.folder_problem[o.name].empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), "%s", s.folder_problem[o.name].c_str());
    }
}

void Title(const char* text) {
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 2.0f);
    ImGui::TextUnformatted(text);
    ImGui::PopFont();
    ImGui::Spacing();
}

bool Back(Screen& s) {
    const bool pressed = ImGui::Button("Back", ImVec2(ImGui::GetFontSize() * 8, 0));
    // the game's B: the button that is not "confirm" with the button layout chosen (Run sets the swap)
    const ImGuiKey pad_back = ImGui::GetIO().ConfigNavSwapGamepadButtons ? ImGuiKey_GamepadFaceDown : ImGuiKey_GamepadFaceRight;
    const bool cancel = !ImGui::IsAnyItemActive() && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopup) &&
                        (ImGui::IsKeyPressed(ImGuiKey_Escape) || ImGui::IsKeyPressed(pad_back));
    if (pressed || cancel) s.page = Page::kMain, s.focus = true;
    return pressed || cancel;
}

std::string MinSec(double secs) {
    const int t = static_cast<int>(secs + 0.5);
    char b[32];
    std::snprintf(b, sizeof b, "%d:%02d", t / 60, t % 60);
    return b;
}

/// (player) Building the game: what is missing, the progress and estimate, Cancel; a failure's reason and its log.
void BuildPage(Screen& s) {
    Title("Build");
    const float em = ImGui::GetFontSize();
    const build::Paths paths = build::Where();
    const build::Status st = build::Get();
    const bool running = build::Running();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Game files: %s", paths.source.empty() ? "not chosen (Settings)" : paths.source.c_str());
    if (paths.packed) ImGui::TextDisabled("Keys: %s", paths.keys.empty() ? "not chosen (Settings)" : paths.keys.c_str());
    ImGui::TextDisabled("Build folder: %s", paths.dir.c_str());
    ImGui::Spacing();
    if (running) {
        ImGui::Text("%s", st.step.c_str());
        const float fraction = st.remaining >= 0 ? static_cast<float>(st.elapsed / (st.elapsed + st.remaining)) : -1.0f;
        char overlay[64] = "";
        if (st.stage == build::Stage::kCompile && st.total) {
            std::snprintf(overlay, sizeof overlay, "%d / %d", st.done, st.total);
        }
        ImGui::ProgressBar(fraction >= 0 ? fraction : -1.0f * static_cast<float>(ImGui::GetTime()), ImVec2(-FLT_MIN, 0),
                           overlay);
        ImGui::Text("%s%s%s", MinSec(st.elapsed).c_str(), st.remaining >= 0 ? " - about " : "",
                    st.remaining >= 0 ? (MinSec(st.remaining) + " left").c_str() : "");
        ImGui::Spacing();
        if (s.focus) ImGui::SetKeyboardFocusHere(), s.focus = false;
        if (ImGui::Button("Cancel", ImVec2(em * 8, 0))) build::Cancel();
    } else {
        // (the last build's outcome only while the folders are the ones it used: a changed setting is a new start)
        const bool same = s.built_for == paths.source + "|" + paths.dir;
        if (same && st.stage == build::Stage::kSeedNext) ImGui::Text("%s", st.step.c_str());
        if (!s.handover_problem.empty()) ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), "%s", s.handover_problem.c_str());
        if (same && st.stage == build::Stage::kDone) ImGui::Text("Built in %s.", MinSec(st.elapsed).c_str());
        if (same && st.stage == build::Stage::kCancelled) ImGui::Text("Cancelled.");
        const bool failed = same && st.stage == build::Stage::kFailed;
        if (failed) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), "%s", st.problem.c_str());
            ImGui::TextDisabled("Log: %s", paths.log.c_str());
        } else if (s.ready) {
            ImGui::Text("Up to date.");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), "%s", s.not_ready.c_str());
        }
        ImGui::Spacing();
        if (s.focus) ImGui::SetKeyboardFocusHere(), s.focus = false;
        if (ImGui::Button(s.ready ? "Rebuild" : "Build", ImVec2(em * 8, 0)) && build::Start(build::Steps::kGame)) {
            s.built_for = paths.source + "|" + paths.dir;
            s.handed_over = false, s.handover_problem.clear();
            Log("start screen: build");
        }
    }
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    Back(s);
}

void MainPage(Screen& s) {
    const float em = ImGui::GetFontSize();
    const ImVec2 button(em * 12, em * 2.2f);
    if (!s.background) {
        Title("Brothership Recomp");
        ImGui::TextDisabled("Mario & Luigi: Brothership, recompiled for PC - beta");
        ImGui::Dummy(ImVec2(0, em * 2));
    } else {
        // the picture carries the title: the buttons at the lower left, on a dark panel so they read over any picture
        const float spacing = ImGui::GetStyle().ItemSpacing.y;
        const int count = s.player ? 5 : 4;
        const float buttons_h = count * button.y + (count - 1) * spacing;
        ImGui::SetCursorPosY(std::max(ImGui::GetCursorPosY(), ImGui::GetWindowHeight() - buttons_h - em * 4));
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(at.x - em, at.y - em),
                                                  ImVec2(at.x + button.x + em, at.y + buttons_h + em),
                                                  IM_COL32(0, 0, 0, 170), em * 0.6f);
    }
    if (s.focus) ImGui::SetKeyboardFocusHere(), s.focus = false;
    if (ImGui::Button("Play", button)) {
        CheckReady(s, true);
        if (s.ready && !build::Running()) s.play = true;
        else s.page = Page::kBuild, s.focus = true; // (not built yet: what is missing, and the Build button)
    }
    if (s.player && ImGui::Button("Build", button)) s.page = Page::kBuild, s.focus = true;
    if (ImGui::Button("Settings", button)) s.page = Page::kSettings, s.focus = true;
    if (ImGui::Button("Mods", button)) s.page = Page::kMods, s.focus = true;
    if (ImGui::Button("Quit", button)) {
        Log("start screen: quit");
        build::Stop();
        Exit(0);
    }
    if (!s.background) {
        ImGui::Dummy(ImVec2(0, em));
        ImGui::TextDisabled("Settings file: %s", options::SettingsPath().c_str());
    }
}

void SettingsPage(Screen& s) {
    Title("Settings");
    if (ImGui::BeginTable("settings", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        for (const options::Option& o : options::All()) {
            if (std::strcmp(o.name, "RT_MODS") == 0) continue; // the mods page
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(o.label);
            ImGui::TableSetColumnIndex(1);
            ImGui::PushID(o.name);
            if (s.focus) ImGui::SetKeyboardFocusHere(), s.focus = false;
            if (FromEnvironment(o)) {
                ImGui::BeginDisabled();
                ImGui::Text("%s", options::Get(o.name));
                ImGui::EndDisabled();
                ImGui::TextDisabled("set by the environment variable %s", o.name);
            } else if (std::strcmp(o.values, "<folder>") == 0) {
                FolderRow(s, o);
            } else if (std::strchr(o.values, '<')) {
                SizeRow(s, o);
            } else {
                ChoiceRow(o);
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    Back(s);
}

void ModsPage(Screen& s) {
    Title("Mods");
    const std::string folder = mods::Folder();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", folder.c_str());
    ImGui::PopTextWrapPos();
    if (ImGui::Button("Open the mods folder")) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(folder), ec);
        SDL_OpenURL(("file:///" + folder).c_str());
    }
    ImGui::Spacing();
    const options::Option* opt = nullptr;
    for (const options::Option& o : options::All()) {
        if (std::strcmp(o.name, "RT_MODS") == 0) opt = &o;
    }
    std::vector<std::string> names; // (not the frame-rate patch: the Frame rate setting switches it, mods.h)
    for (const std::string& n : s.built) {
        if (n != mods::kFrameRateMod) names.push_back(n);
    }
    for (const std::string& f : mods::Folders()) {
        if (f != mods::kFrameRateMod && std::find(names.begin(), names.end(), f) == names.end()) names.push_back(f);
    }
    std::sort(names.begin(), names.end());
    if (FromEnvironment(*opt)) {
        ImGui::TextDisabled("Set by the environment variable RT_MODS=%s", options::Get("RT_MODS"));
    } else if (names.empty()) {
        ImGui::TextDisabled("No mods found.");
    } else {
        const std::string saved = options::Saved("RT_MODS");
        std::vector<std::string> on = Split(saved, ',');
        if (saved == "all") on = names;
        if (saved == "none") on.clear();
        for (const std::string& name : names) {
            const bool in_build = std::find(s.built.begin(), s.built.end(), name) != s.built.end();
            const bool needs_lift = !in_build && mods::HasExefs(name);
            bool checked = !needs_lift && std::find(on.begin(), on.end(), name) != on.end();
            if (s.focus) ImGui::SetKeyboardFocusHere(), s.focus = false;
            ImGui::BeginDisabled(needs_lift);
            if (ImGui::Checkbox(name.c_str(), &checked)) {
                std::vector<std::string> next;
                for (const std::string& n : names) {
                    const bool was = std::find(on.begin(), on.end(), n) != on.end();
                    if (n == name ? checked : was) next.push_back(n);
                }
                std::string value;
                for (const std::string& n : next) value += (value.empty() ? "" : ",") + n;
                Saved("RT_MODS", value);
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (needs_lift) {
                ImGui::TextDisabled(s.player ? "(code patch: build the game again to use it)"
                                             : "(patches the game's code: needs a build with it built in)");
            } else {
                ImGui::TextDisabled("%s%s%s", in_build ? "code patch" : "", in_build && mods::HasRomfs(name) ? ", " : "",
                                    mods::HasRomfs(name) ? "game files" : "");
            }
        }
    }
    ImGui::Spacing();
    Back(s);
}

PFN_vkVoidFunction LoadVulkan(const char* name, void*) {
    nvn::VulkanCore& v = nvn::Vk();
    if (PFN_vkVoidFunction f = v.dld.vkGetDeviceProcAddr(*v.device->GetLogical(), name)) return f;
    return v.dld.vkGetInstanceProcAddr(*v.instance, name);
}

} // namespace

void ApplyStyle() {
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.FontSizeBase = 20.0f;
    style.WindowRounding = 0.0f;
    style.FrameRounding = 4.0f;
    style.ItemSpacing.y = 8.0f;
}

std::string BackgroundPath() {
    const char* named = std::getenv("RT_UI_BACKGROUND");
    std::error_code ec;
    std::string path;
    if (named && *named) {
        path = named;
    } else {
        const std::filesystem::path mine = std::filesystem::path(options::UserDir()) / "background.png";
        const std::filesystem::path shipped =
            std::filesystem::path(platform::ExecutablePath()).parent_path() / "background.png";
        path = (std::filesystem::exists(mine, ec) ? mine : shipped).string();
    }
    if (!std::filesystem::exists(std::filesystem::path(path), ec)) {
        if (named && *named) Log("start screen: no background picture %s", path.c_str());
        return "";
    }
    return path;
}

ImTextureData* LoadBackground() {
    const std::string path = BackgroundPath();
    if (path.empty()) return nullptr;
    SDL_Surface* loaded = SDL_LoadPNG(path.c_str());
    SDL_Surface* rgba = loaded ? SDL_ConvertSurface(loaded, SDL_PIXELFORMAT_RGBA32) : nullptr;
    if (loaded) SDL_DestroySurface(loaded);
    if (!rgba) {
        Log("start screen: background %s not loaded: %s", path.c_str(), SDL_GetError());
        return nullptr;
    }
    auto* tex = new ImTextureData;
    tex->Create(ImTextureFormat_RGBA32, rgba->w, rgba->h);
    tex->UseColors = true;
    for (int y = 0; y < rgba->h; ++y) {
        std::memcpy(tex->GetPixelsAt(0, y), static_cast<const unsigned char*>(rgba->pixels) + y * rgba->pitch,
                    static_cast<size_t>(rgba->w) * 4);
    }
    Log("start screen: background %s (%dx%d)", path.c_str(), rgba->w, rgba->h);
    SDL_DestroySurface(rgba);
    ImGui::RegisterUserTexture(tex);
    return tex;
}

void DrawBackground(ImTextureData* background, float dim) {
    if (!background) return;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float vw = vp->Size.x, vh = vp->Size.y;
    const float iw = static_cast<float>(background->Width), ih = static_cast<float>(background->Height);
    ImVec2 uv0(0, 0), uv1(1, 1);
    if (vw / vh > iw / ih) { // the window is wider: crop top and bottom
        const float keep = (iw / ih) / (vw / vh);
        uv0.y = (1 - keep) / 2, uv1.y = 1 - uv0.y;
    } else {
        const float keep = (vw / vh) / (iw / ih);
        uv0.x = (1 - keep) / 2, uv1.x = 1 - uv0.x;
    }
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImVec2 p0 = vp->Pos, p1(vp->Pos.x + vw, vp->Pos.y + vh);
    dl->AddImage(background->GetTexRef(), p0, p1, uv0, uv1);
    if (dim > 0) dl->AddRectFilled(p0, p1, IM_COL32(0, 0, 0, static_cast<int>(dim * 255)));
}

bool Wanted(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--play") == 0) return false;
    }
    const char* skip = std::getenv("RT_NO_LAUNCHER");
    const char* input = std::getenv("RT_INPUT");
    return window::IsOpen() && nvn::Enabled() && !(skip && *skip && std::strcmp(skip, "0") != 0) && !(input && *input);
}

void Run(const std::vector<std::string>& built_mods, bool player, bool build_page) {
    Screen s;
    s.built = built_mods;
    s.player = player;
    if (build_page) s.page = Page::kBuild; // (back from `--seed`)
    for (const options::Option& o : options::All()) {
        if (std::strcmp(o.values, "<folder>") == 0) {
            std::snprintf(s.folder[o.name].data(), s.folder[o.name].size(), "%s", options::Saved(o.name).c_str());
        }
    }
    CheckReady(s, true);
    std::snprintf(s.size, sizeof s.size, "%s", options::Saved("RT_WINDOW_SIZE").c_str());
    SDL_Window* w = window::Sdl();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.IniFilename = nullptr; // nothing of ImGui's own on disk
    io.Fonts->AddFontDefaultVector();
    ApplyStyle();
    ImGuiStyle& style = ImGui::GetStyle();
    s.background = LoadBackground();
    if (s.background) style.Colors[ImGuiCol_WindowBg].w = 0.0f; // the picture shows through the page

    VkRenderPass pass = VK_NULL_HANDLE;
    u32 images = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    while (!nvn::UiTarget(&pass, &images, &format)) { // (minimised at start: wait for an area)
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) Exit(0);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    nvn::VulkanCore& v = nvn::Vk();
    const auto& dev = *v.device;
    const u32 api = dev.ApiVersion();
    if (!ImGui_ImplVulkan_LoadFunctions(api, LoadVulkan)) Fatal("start screen: ImGui could not load Vulkan's functions");
    VkQueue queue = VK_NULL_HANDLE;
    reinterpret_cast<PFN_vkGetDeviceQueue>(LoadVulkan("vkGetDeviceQueue", nullptr))(
        *dev.GetLogical(), dev.GetGraphicsFamily(), 0, &queue);
    ImGui_ImplSDL3_InitForVulkan(w);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = api;
    info.Instance = *v.instance;
    info.PhysicalDevice = dev.GetPhysical();
    info.Device = *dev.GetLogical();
    info.QueueFamily = dev.GetGraphicsFamily();
    info.Queue = queue;
    info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE + 8;
    info.MinImageCount = 2;
    info.ImageCount = std::max<u32>(images, 2) + 1; // (vertex buffers rotate per image: the frames in flight too)
    info.PipelineInfoMain.RenderPass = pass;
    info.PipelineInfoMain.Subpass = 0;
    if (!ImGui_ImplVulkan_Init(&info)) Fatal("start screen: ImGui's Vulkan backend did not start");
    Log("start screen: shown (ImGui %s; settings file %s)", IMGUI_VERSION, options::SettingsPath().c_str());

    auto next = std::chrono::steady_clock::now();
    // RT_UI_TEST (tests, comma-separated): play@<seconds> - Play by itself after that long (the hand-over to the
    // game); page=settings|mods - open on that page (screenshots); set:<option>=<value> - save a setting from the
    // screen, as a click does (after the start: what was read before Play must not count)
    double autoplay = -1;
    if (const char* t = std::getenv("RT_UI_TEST")) {
        for (const std::string& part : Split(t, ',')) {
            std::sscanf(part.c_str(), "play@%lf", &autoplay);
            if (part == "page=settings") s.page = Page::kSettings;
            if (part == "page=mods") s.page = Page::kMods;
            if (part == "page=build") s.page = Page::kBuild;
            // (a test: Build pressed at once - not again when back from `--seed`, which inherits RT_UI_TEST)
            if (part == "build" && s.player && !build_page && build::Start(build::Steps::kGame)) {
                const build::Paths paths = build::Where();
                s.built_for = paths.source + "|" + paths.dir;
            }
            if (part.rfind("set:", 0) == 0 && part.find('=') != std::string::npos) { // as a click on the page saves it
                const size_t eq = part.find('=');
                Saved(part.substr(4, eq - 4).c_str(), part.substr(eq + 1));
            }
        }
    }
    const auto shown = next;
    while (!s.play) {
        CheckReady(s);
        if (s.player && !s.handed_over && !build::Running() && build::Get().stage == build::Stage::kSeedNext) HandOver(s);
        if (autoplay >= 0 && !build::Running() && s.ready &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - shown).count() >= autoplay) {
            Log("start screen: RT_UI_TEST: play");
            s.play = true;
            break;
        }
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL3_ProcessEvent(&e);
            if (e.type == SDL_EVENT_GAMEPAD_ADDED || e.type == SDL_EVENT_GAMEPAD_REMOVED) input::OnEvent(e);
            if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
                Log("start screen: window closed");
                build::Stop();
                Exit(0);
            }
        }
        if (SDL_GetWindowFlags(w) & SDL_WINDOW_MINIMIZED) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        int w_px = 1280, h_px = 720;
        SDL_GetWindowSizeInPixels(w, &w_px, &h_px);
        style.FontScaleMain = std::max(1.0f, h_px / 720.0f); // the same look at 720p, 1440p and on the Deck

        // confirm / back as in the game: ImGui's confirm is the bottom button unless swapped (RT_BUTTONS)
        io.ConfigNavSwapGamepadButtons = !input::BottomConfirms();
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        DrawBackground(s.background, s.page == Page::kMain ? 0.0f : 0.72f);
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        const float pad = ImGui::GetFontSize() * 3;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(pad, pad));
        ImGui::Begin("start", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                         ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PopStyleVar();
        switch (s.page) {
        case Page::kMain: MainPage(s); break;
        case Page::kSettings: SettingsPage(s); break;
        case Page::kMods: ModsPage(s); break;
        case Page::kBuild: BuildPage(s); break;
        }
        ImGui::End();
        ImGui::Render();
        VkExtent2D extent{};
        nvn::ShowUi([](VkCommandBuffer cmd) { ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd); }, &extent);
        // ~60 frames a second is plenty for a menu (mailbox would otherwise draw as fast as the GPU can)
        next += std::chrono::milliseconds(16);
        const auto now = std::chrono::steady_clock::now();
        if (next < now) next = now;
        std::this_thread::sleep_until(next);
    }
    Log("start screen: play");
    nvn::EndUi();
    {
        std::lock_guard queue_lock(nvn::QueueLock()); // (its shutdown waits for the device)
        ImGui_ImplVulkan_Shutdown();                   // (and frees the textures it made, the background's too)
    }
    if (s.background) {
        ImGui::UnregisterUserTexture(s.background);
        delete s.background;
    }
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
}

} // namespace rt::ui
