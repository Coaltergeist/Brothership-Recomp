// SDL3 window for video_core, modelled on drippu's
// src/suyu_cmd/emu_window/emu_window_sdl2_vk.cpp (SDL3 despite the name).
#include "window.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

#include <SDL3/SDL.h>

#include "input.h"
#include "options.h"
#include "perf.h"
#include "resolution.h"

#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"

namespace rt::window {
namespace {

constexpr int kWidth = 1280, kHeight = 720; // the Switch's handheld resolution

class NoContext final : public Core::Frontend::GraphicsContext {};

/// No window: WindowSystemInfo stays Headless, which RendererVulkan handles
/// without a surface (frames are read back into memory).
class HeadlessWindow final : public Core::Frontend::EmuWindow {
public:
    HeadlessWindow() { UpdateCurrentFramebufferLayout(kWidth, kHeight); }
    std::unique_ptr<Core::Frontend::GraphicsContext> CreateSharedContext() const override {
        return std::make_unique<NoContext>();
    }
    bool IsShown() const override { return true; }
};

class SdlWindow final : public Core::Frontend::EmuWindow {
public:
    explicit SdlWindow(SDL_Window* w) : window_{w} {
        const SDL_PropertiesID props = SDL_GetWindowProperties(w);
#if defined(SDL_PLATFORM_WIN32)
        window_info.type = Core::Frontend::WindowSystemType::Windows;
        window_info.render_surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
#elif defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_FREEBSD)
        const char* driver = SDL_GetCurrentVideoDriver();
        if (driver && std::strcmp(driver, "x11") == 0) {
            window_info.type = Core::Frontend::WindowSystemType::X11;
            window_info.display_connection = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
            window_info.render_surface =
                reinterpret_cast<void*>(SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
        } else if (driver && std::strcmp(driver, "wayland") == 0) {
            window_info.type = Core::Frontend::WindowSystemType::Wayland;
            window_info.display_connection =
                SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr);
            window_info.render_surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
        } else {
            Fatal("window: unsupported SDL video driver %s", driver ? driver : "(none)");
        }
#else
        Fatal("window: no Vulkan surface for this platform yet");
#endif
        if (!window_info.render_surface) Fatal("window: no native window handle: %s", SDL_GetError());
        int w_px = kWidth, h_px = kHeight;
        SDL_GetWindowSizeInPixels(w, &w_px, &h_px);
        UpdateCurrentFramebufferLayout(static_cast<u32>(w_px), static_cast<u32>(h_px));
    }
    std::unique_ptr<Core::Frontend::GraphicsContext> CreateSharedContext() const override {
        return std::make_unique<NoContext>();
    }
    bool IsShown() const override { return !minimized.load(std::memory_order_relaxed); }

    SDL_Window* Get() const { return window_; }
    std::atomic<bool> minimized{false};

private:
    SDL_Window* window_;
};

std::unique_ptr<SdlWindow> g_window;
HeadlessWindow* g_headless = nullptr;

void UpdateTitle(SDL_Window* w) {
    const perf::Second s = perf::LastSecond();
    char title[128];
    if (perf::Frames() == 0) {
        std::snprintf(title, sizeof title, "nxrt");
    } else {
        std::snprintf(title, sizeof title, "nxrt | %u fps | frame time p50 %.1f / p95 %.1f / max %.1f ms", s.frames,
                      s.p50_ms, s.p95_ms, s.max_ms);
    }
    SDL_SetWindowTitle(w, title);
}

} // namespace

bool Open() {
    const char* h = std::getenv("RT_HEADLESS");
    if (h && std::strcmp(h, "0") != 0) {
        g_headless = new HeadlessWindow;
        resolution::Choose(0);
        return false;
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        Fatal("window: SDL_Init failed: %s (RT_HEADLESS=1 runs without a window)", SDL_GetError());
    }
    // RT_WINDOW=windowed (default) | borderless (SDL3's desktop full screen: no frame, the display's size);
    // RT_WINDOW_SIZE=<w>x<h> the windowed size. The game still renders at 720p; the presenter scales.
    const char* mode = options::Get("RT_WINDOW");
    const bool borderless = mode && std::strcmp(mode, "borderless") == 0;
    if (mode && !borderless && std::strcmp(mode, "windowed") != 0) Fatal("window: RT_WINDOW=%s (windowed|borderless)", mode);
    int width = kWidth, height = kHeight;
    if (const char* size = options::Get("RT_WINDOW_SIZE")) {
        if (std::sscanf(size, "%dx%d", &width, &height) != 2 || width <= 0 || height <= 0) {
            Fatal("window: RT_WINDOW_SIZE=%s (expected <width>x<height>)", size);
        }
    }
    SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    if (borderless) flags |= SDL_WINDOW_FULLSCREEN;
    SDL_Window* w = SDL_CreateWindow("nxrt", width, height, flags);
    if (!w) Fatal("window: SDL_CreateWindow failed: %s", SDL_GetError());
    input::Init();
    g_window = std::make_unique<SdlWindow>(w);
    SDL_ShowWindow(w);
    SDL_RaiseWindow(w);
    SDL_PumpEvents();
    int w_px = width, h_px = height;
    SDL_GetWindowSizeInPixels(w, &w_px, &h_px);
    resolution::Choose(h_px);
    Log("window: %dx%d pixels%s, SDL %d.%d.%d, video driver %s", w_px, h_px, borderless ? " (borderless)" : "",
        SDL_VERSIONNUM_MAJOR(SDL_GetVersion()), SDL_VERSIONNUM_MINOR(SDL_GetVersion()),
        SDL_VERSIONNUM_MICRO(SDL_GetVersion()), SDL_GetCurrentVideoDriver());
    return true;
}

bool IsOpen() {
    return g_window != nullptr;
}

Core::Frontend::EmuWindow& Emu() {
    if (g_window) return *g_window;
    if (!g_headless) g_headless = new HeadlessWindow; // Open() was not called (tools, tests)
    return *g_headless;
}

void RunEventLoop() {
    if (!g_window) {
        for (;;) std::this_thread::sleep_for(std::chrono::hours(1)); // the guest ends the process
    }
    SDL_Window* w = g_window->Get();
    auto next_title = std::chrono::steady_clock::now();
    for (;;) {
        SDL_Event e;
        if (SDL_WaitEventTimeout(&e, 250)) {
            do {
                input::OnEvent(e);
                switch (e.type) {
                case SDL_EVENT_QUIT:
                case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                    Log("window: closed");
                    Exit(0);
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                    g_window->UpdateCurrentFramebufferLayout(static_cast<u32>(e.window.data1),
                                                             static_cast<u32>(e.window.data2));
                    break;
                case SDL_EVENT_WINDOW_MINIMIZED:
                    g_window->minimized.store(true, std::memory_order_relaxed);
                    break;
                case SDL_EVENT_WINDOW_RESTORED:
                case SDL_EVENT_WINDOW_MAXIMIZED:
                    g_window->minimized.store(false, std::memory_order_relaxed);
                    break;
                default:
                    break;
                }
            } while (SDL_PollEvent(&e));
        }
        if (std::chrono::steady_clock::now() >= next_title) {
            next_title += std::chrono::seconds(1);
            UpdateTitle(w);
        }
    }
}

} // namespace rt::window
