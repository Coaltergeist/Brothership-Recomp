// The shader cache's own run (seed.h).
#include "seed.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include <SDL3/SDL.h>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#include "../build/build.h"
#include "../options.h"
#include "../platform/platform.h"
#include "../rt.h"
#include "look.h"

namespace rt::ui {
namespace {

std::string MinSec(double secs) {
    const int t = static_cast<int>(secs + 0.5);
    char b[32];
    std::snprintf(b, sizeof b, "%d:%02d", t / 60, t % 60);
    return b;
}

bool Finished(build::Stage s) {
    return s == build::Stage::kDone || s == build::Stage::kFailed || s == build::Stage::kCancelled;
}

/// The window as the start screen's (window.cpp: RT_WINDOW borderless = SDL's desktop full screen, else
/// RT_WINDOW_SIZE), nullptr when none can be made.
SDL_Window* OpenWindow() {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        Log("seed: no window (%s): building without one", SDL_GetError());
        return nullptr;
    }
    const char* mode = options::Get("RT_WINDOW");
    int width = 1280, height = 720;
    if (const char* size = options::Get("RT_WINDOW_SIZE")) std::sscanf(size, "%dx%d", &width, &height);
    SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    if (mode && std::strcmp(mode, "borderless") == 0) flags |= SDL_WINDOW_FULLSCREEN;
    SDL_Window* w = SDL_CreateWindow("Brothership Recomp", std::max(width, 320), std::max(height, 180), flags);
    if (!w) Log("seed: no window (%s): building without one", SDL_GetError());
    return w;
}

/// The background picture for SDL's software renderer: cropped to the window's aspect ratio (centred) and scaled to its
/// size once, drawn with SDL_RenderTexture - as an ImGui image, scaled through the software renderer's triangles, it
/// showed nothing (2026-11-01), and one copy a frame costs less CPU beside the seed. nullptr: none.
SDL_Texture* BackgroundTexture(SDL_Renderer* r, int w_px, int h_px) {
    const std::string path = BackgroundPath();
    if (path.empty()) return nullptr;
    SDL_Surface* loaded = SDL_LoadPNG(path.c_str());
    SDL_Surface* rgba = loaded ? SDL_ConvertSurface(loaded, SDL_PIXELFORMAT_RGBA32) : nullptr;
    if (loaded) SDL_DestroySurface(loaded);
    if (!rgba) {
        Log("seed: background %s not loaded: %s", path.c_str(), SDL_GetError());
        return nullptr;
    }
    SDL_Rect crop{0, 0, rgba->w, rgba->h};
    if (static_cast<float>(w_px) / h_px > static_cast<float>(rgba->w) / rgba->h) { // the window is wider: top and bottom
        crop.h = static_cast<int>(static_cast<float>(rgba->w) * h_px / w_px);
        crop.y = (rgba->h - crop.h) / 2;
    } else {
        crop.w = static_cast<int>(static_cast<float>(rgba->h) * w_px / h_px);
        crop.x = (rgba->w - crop.w) / 2;
    }
    SDL_Surface* sized = SDL_CreateSurface(w_px, h_px, SDL_PIXELFORMAT_RGBA32);
    SDL_Texture* t = nullptr;
    if (sized && SDL_BlitSurfaceScaled(rgba, &crop, sized, nullptr, SDL_SCALEMODE_LINEAR)) t = SDL_CreateTextureFromSurface(r, sized);
    if (!t) Log("seed: background %s not drawn: %s", path.c_str(), SDL_GetError());
    else Log("seed: background %s (%dx%d, scaled to %dx%d)", path.c_str(), rgba->w, rgba->h, w_px, h_px);
    if (sized) SDL_DestroySurface(sized);
    SDL_DestroySurface(rgba);
    return t;
}

/// The steps without a window: waited for (the build's own log has their progress).
int WithoutWindow() {
    while (build::Running() || !Finished(build::Get().stage)) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return build::Get().stage == build::Stage::kDone ? 0 : 1;
}

/// The page: the progress while running, the outcome after a failure; true when Back was chosen.
bool Page(const build::Status& st, bool running) {
    const float em = ImGui::GetFontSize();
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 2.0f);
    ImGui::TextUnformatted("Build");
    ImGui::PopFont();
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    bool back = false;
    if (running || !Finished(st.stage)) {
        ImGui::Text("%s", st.step.empty() ? "Building the shader cache" : st.step.c_str());
        const float fraction = st.remaining >= 0 && st.elapsed + st.remaining > 0
                                   ? static_cast<float>(st.elapsed / (st.elapsed + st.remaining))
                                   : -1.0f * static_cast<float>(ImGui::GetTime());
        char overlay[64] = "";
        if (st.stage == build::Stage::kSeed && st.total) std::snprintf(overlay, sizeof overlay, "%d / %d", st.done, st.total);
        ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0), overlay);
        ImGui::Text("%s%s%s", MinSec(st.elapsed).c_str(), st.remaining >= 0 ? " - about " : "",
                    st.remaining >= 0 ? (MinSec(st.remaining) + " left").c_str() : "");
        ImGui::Spacing();
        if (ImGui::Button("Cancel", ImVec2(em * 8, 0))) build::Cancel();
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), "%s", st.problem.c_str());
        ImGui::TextDisabled("Log: %s", build::Where().log.c_str());
        ImGui::Spacing();
        back = ImGui::Button("Back", ImVec2(em * 8, 0));
    }
    ImGui::PopTextWrapPos();
    return back;
}

} // namespace

bool StartSelf(const char* arg, std::string* why) {
    const std::string self = platform::ExecutablePath();
    const char* args[] = {self.c_str(), arg && *arg ? arg : nullptr, nullptr};
    const SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, args);
    SDL_SetBooleanProperty(props, SDL_PROP_PROCESS_CREATE_BACKGROUND_BOOLEAN, true); // (outlives us)
    SDL_Process* p = SDL_CreateProcessWithProperties(props);
    SDL_DestroyProperties(props);
    if (!p) {
        *why = SDL_GetError();
        return false;
    }
    Log("start again: %s %s", self.c_str(), arg && *arg ? arg : "");
    return true;
}

int RunSeed() {
    Log("seed: the shader cache, in a run of its own (no Vulkan device here)");
    if (!build::Start(build::Steps::kShaders)) return 1;
    SDL_Window* w = OpenWindow();
    SDL_Renderer* r = w ? SDL_CreateRenderer(w, SDL_SOFTWARE_RENDERER) : nullptr; // (no GPU API in this process)
    if (w && !r) Log("seed: no software renderer (%s): building without a window", SDL_GetError());
    int code = 0;
    if (!r) {
        if (w) SDL_DestroyWindow(w);
        code = WithoutWindow();
    } else {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
        io.IniFilename = nullptr;
        io.Fonts->AddFontDefaultVector();
        ApplyStyle();
        ImGuiStyle& style = ImGui::GetStyle();
        ImGui_ImplSDL3_InitForSDLRenderer(w, r);
        ImGui_ImplSDLRenderer3_Init(r);
        SDL_ShowWindow(w);
        SDL_RaiseWindow(w);
        SDL_SyncWindow(w); // (borderless: the display's size before the picture is scaled to it)
        SDL_Texture* background = nullptr;
        int bg_w = 0, bg_h = 0;
        auto next = std::chrono::steady_clock::now();
        for (bool back = false; !back;) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                ImGui_ImplSDL3_ProcessEvent(&e);
                if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
                    Log("seed: window closed: cancelling");
                    build::Stop();
                    back = true;
                }
            }
            const build::Status st = build::Get();
            const bool running = build::Running();
            if (!running && (st.stage == build::Stage::kDone || st.stage == build::Stage::kCancelled)) break;
            int w_px = 1280, h_px = 720;
            SDL_GetWindowSizeInPixels(w, &w_px, &h_px);
            style.FontScaleMain = std::max(1.0f, h_px / 720.0f);
            if (w_px != bg_w || h_px != bg_h) { // (made again only when the window's size changes)
                if (background) SDL_DestroyTexture(background);
                background = w_px > 0 && h_px > 0 ? BackgroundTexture(r, w_px, h_px) : nullptr;
                bg_w = w_px, bg_h = h_px;
                style.Colors[ImGuiCol_WindowBg].w = background ? 0.0f : 1.0f; // the picture shows through the page
            }
            ImGui_ImplSDLRenderer3_NewFrame();
            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();
            const ImGuiViewport* vp = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(vp->WorkPos);
            ImGui::SetNextWindowSize(vp->WorkSize);
            const float pad = ImGui::GetFontSize() * 3;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(pad, pad));
            ImGui::Begin("seed", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoBringToFrontOnFocus);
            ImGui::PopStyleVar();
            if (Page(st, running)) back = true;
            ImGui::End();
            ImGui::Render();
            SDL_SetRenderDrawColor(r, 0, 0, 0, 255);
            SDL_RenderClear(r);
            if (background) { // darkened as the start screen's pages (0.72)
                SDL_RenderTexture(r, background, nullptr, nullptr);
                SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
                SDL_SetRenderDrawColor(r, 0, 0, 0, static_cast<Uint8>(0.72f * 255));
                SDL_RenderFillRect(r, nullptr);
            }
            ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), r);
            SDL_RenderPresent(r);
            // 30 frames a second: drawn on the CPU, beside the seed's own threads
            next += std::chrono::milliseconds(33);
            const auto now = std::chrono::steady_clock::now();
            if (next < now) next = now;
            std::this_thread::sleep_until(next);
        }
        code = build::Get().stage == build::Stage::kDone ? 0 : 1;
        ImGui_ImplSDLRenderer3_Shutdown(); // (frees the textures it made)
        if (background) SDL_DestroyTexture(background);
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        SDL_DestroyRenderer(r);
        SDL_DestroyWindow(w);
    }
    Log("seed: %s", build::StageName(build::Get().stage));
    std::string why;
    if (!StartSelf("--after-seed", &why)) Log("seed: could not start the start screen again: %s", why.c_str());
    return code;
}

} // namespace rt::ui
