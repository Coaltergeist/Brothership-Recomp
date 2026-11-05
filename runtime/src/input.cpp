#include "input.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#include <SDL3/SDL.h>

#include "hle.h"
#include "options.h"

namespace rt::input {
namespace {

// HidNpadButton bits (libnx; hid.cpp's names).
constexpr u64 kA = 1u << 0, kB = 1u << 1, kX = 1u << 2, kY = 1u << 3, kStickL = 1u << 4, kStickR = 1u << 5,
              kL = 1u << 6, kR = 1u << 7, kZL = 1u << 8, kZR = 1u << 9, kPlus = 1u << 10, kMinus = 1u << 11,
              kLeft = 1u << 12, kUp = 1u << 13, kRight = 1u << 14, kDown = 1u << 15;
constexpr s32 kStickMax = 0x7FFF;

constexpr struct {
    SDL_Scancode key;
    u64 bit;
} kKeys[] = {{SDL_SCANCODE_Z, kA},         {SDL_SCANCODE_X, kB},          {SDL_SCANCODE_C, kX},
             {SDL_SCANCODE_V, kY},         {SDL_SCANCODE_E, kL},          {SDL_SCANCODE_U, kR},
             {SDL_SCANCODE_Q, kZL},        {SDL_SCANCODE_O, kZR},         {SDL_SCANCODE_RETURN, kPlus},
             {SDL_SCANCODE_BACKSPACE, kMinus}, {SDL_SCANCODE_F, kStickL}, {SDL_SCANCODE_N, kStickR},
             {SDL_SCANCODE_LEFT, kLeft},   {SDL_SCANCODE_UP, kUp},        {SDL_SCANCODE_RIGHT, kRight},
             {SDL_SCANCODE_DOWN, kDown}};

// The face buttons: RT_BUTTONS=positions (default) as on a Switch - SDL's face buttons are positional (south =
// bottom), so an Xbox pad's B is A -, or labels: the button the controller has printed A (Xbox, PlayStation's cross)
// is A, from SDL's label for each controller (a Nintendo layout pad is the same either way).
constexpr SDL_GamepadButton kFaceButtons[] = {SDL_GAMEPAD_BUTTON_SOUTH, SDL_GAMEPAD_BUTTON_EAST, SDL_GAMEPAD_BUTTON_WEST,
                                              SDL_GAMEPAD_BUTTON_NORTH};

bool ByLabel() {
    const char* v = options::Get("RT_BUTTONS");
    return v && std::strcmp(v, "labels") == 0;
}

u64 FaceBit(SDL_Gamepad* g, SDL_GamepadButton b, bool by_label) {
    if (by_label) {
        switch (SDL_GetGamepadButtonLabel(g, b)) {
        case SDL_GAMEPAD_BUTTON_LABEL_A:
        case SDL_GAMEPAD_BUTTON_LABEL_CROSS: return kA;
        case SDL_GAMEPAD_BUTTON_LABEL_B:
        case SDL_GAMEPAD_BUTTON_LABEL_CIRCLE: return kB;
        case SDL_GAMEPAD_BUTTON_LABEL_X:
        case SDL_GAMEPAD_BUTTON_LABEL_SQUARE: return kX;
        case SDL_GAMEPAD_BUTTON_LABEL_Y:
        case SDL_GAMEPAD_BUTTON_LABEL_TRIANGLE: return kY;
        default: break; // (no label known: by position)
        }
    }
    return b == SDL_GAMEPAD_BUTTON_EAST ? kA : b == SDL_GAMEPAD_BUTTON_SOUTH ? kB : b == SDL_GAMEPAD_BUTTON_NORTH ? kX : kY;
}

constexpr struct {
    SDL_GamepadButton button;
    u64 bit;
} kGamepadButtons[] = {{SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, kL},  {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, kR},
                       {SDL_GAMEPAD_BUTTON_START, kPlus},       {SDL_GAMEPAD_BUTTON_BACK, kMinus},
                       {SDL_GAMEPAD_BUTTON_LEFT_STICK, kStickL}, {SDL_GAMEPAD_BUTTON_RIGHT_STICK, kStickR},
                       {SDL_GAMEPAD_BUTTON_DPAD_LEFT, kLeft},   {SDL_GAMEPAD_BUTTON_DPAD_UP, kUp},
                       {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, kRight}, {SDL_GAMEPAD_BUTTON_DPAD_DOWN, kDown}};

std::mutex g_mu;
Pad g_pad;                                  // under g_mu
std::atomic<bool> g_capture_key{false};     // F9 pressed (TakeCaptureKey)
std::atomic<bool> g_toggle_key{false};      // F10 pressed (TakeToggleKey)
std::atomic<bool> g_fps_key{false};         // F8 pressed (TakeFpsKey)
std::atomic<bool> g_flag_key{false};        // F7 or a controller's Capture/Share button pressed (TakeFlagKey)
std::map<SDL_JoystickID, SDL_Gamepad*> g_pads; // main thread only

/// A key pair as one axis: full deflection, scaled onto the circle when both axes are pressed.
void KeyStick(const bool* keys, SDL_Scancode neg_x, SDL_Scancode pos_x, SDL_Scancode neg_y, SDL_Scancode pos_y,
              s32& x, s32& y) {
    const int dx = keys[pos_x] - keys[neg_x], dy = keys[pos_y] - keys[neg_y];
    const s32 v = dx && dy ? 23170 : kStickMax; // 0x7FFF / sqrt(2)
    x = dx * v, y = dy * v;
}

/// A gamepad stick (SDL: -32768..32767, down positive) with a radial dead zone, up positive.
void PadStick(SDL_Gamepad* g, SDL_GamepadAxis ax, SDL_GamepadAxis ay, s32& x, s32& y) {
    const double fx = SDL_GetGamepadAxis(g, ax) / 32767.0, fy = -SDL_GetGamepadAxis(g, ay) / 32767.0;
    const double m = std::hypot(fx, fy);
    constexpr double kDeadZone = 0.10; // Ryujinx's default; a resting pad reads a few hundred, not 0
    if (m < kDeadZone) {
        x = y = 0;
        return;
    }
    const double scale = std::min(1.0, (m - kDeadZone) / (1 - kDeadZone)) / m;
    x = static_cast<s32>(std::lround(std::clamp(fx * scale, -1.0, 1.0) * kStickMax));
    y = static_cast<s32>(std::lround(std::clamp(fy * scale, -1.0, 1.0) * kStickMax));
}

/// Keeps the stick that is pushed further.
void Stronger(s32& x, s32& y, s32 nx, s32 ny) {
    if (s64{nx} * nx + s64{ny} * ny > s64{x} * x + s64{y} * y) x = nx, y = ny;
}

/// The state of every device now (main thread).
Pad Poll() {
    Pad p;
    const bool* keys = SDL_GetKeyboardState(nullptr);
    for (const auto& k : kKeys) {
        if (keys[k.key]) p.buttons |= k.bit;
    }
    KeyStick(keys, SDL_SCANCODE_A, SDL_SCANCODE_D, SDL_SCANCODE_S, SDL_SCANCODE_W, p.lx, p.ly);
    KeyStick(keys, SDL_SCANCODE_J, SDL_SCANCODE_L, SDL_SCANCODE_K, SDL_SCANCODE_I, p.rx, p.ry);
    const bool by_label = ByLabel();
    for (const auto& [id, g] : g_pads) {
        for (const auto& b : kGamepadButtons) {
            if (SDL_GetGamepadButton(g, b.button)) p.buttons |= b.bit;
        }
        for (SDL_GamepadButton b : kFaceButtons) {
            if (SDL_GetGamepadButton(g, b)) p.buttons |= FaceBit(g, b, by_label);
        }
        constexpr Sint16 kTrigger = 0x3000; // a trigger counts as pressed past ~40 %
        if (SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > kTrigger) p.buttons |= kZL;
        if (SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > kTrigger) p.buttons |= kZR;
        s32 x, y;
        PadStick(g, SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_LEFTY, x, y);
        Stronger(p.lx, p.ly, x, y);
        PadStick(g, SDL_GAMEPAD_AXIS_RIGHTX, SDL_GAMEPAD_AXIS_RIGHTY, x, y);
        Stronger(p.rx, p.ry, x, y);
    }
    return p;
}

void Open(SDL_JoystickID id) {
    if (g_pads.count(id)) return;
    if (SDL_Gamepad* g = SDL_OpenGamepad(id)) {
        g_pads[id] = g;
        Log("input: controller connected: %s", SDL_GetGamepadName(g));
    }
}

} // namespace

bool BottomConfirms() {
    const bool by_label = ByLabel();
    if (g_pads.empty()) return by_label; // (no controller yet: an Xbox-style pad by label)
    return FaceBit(g_pads.begin()->second, SDL_GAMEPAD_BUTTON_SOUTH, by_label) == kA;
}

void Init() {
    if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
        Log("input: no game controllers (SDL_INIT_GAMEPAD failed: %s); keyboard only", SDL_GetError());
        return;
    }
    // Mappings for controllers SDL does not know (SDL's gamecontrollerdb.txt format), from the working directory.
    if (SDL_GetPathInfo("gamecontrollerdb.txt", nullptr)) {
        const int n = SDL_AddGamepadMappingsFromFile("gamecontrollerdb.txt");
        if (n < 0) Log("input: gamecontrollerdb.txt: %s", SDL_GetError());
        else Log("input: gamecontrollerdb.txt: %d controller mapping(s) added", n);
    }
    // Controllers connected before start: opened now, not left to their ADDED events (as LASRecomp does).
    int n = 0;
    if (SDL_JoystickID* ids = SDL_GetGamepads(&n)) {
        for (int i = 0; i < n; ++i) Open(ids[i]);
        SDL_free(ids);
    }
    if (n == 0) Log("input: no controller connected (keyboard: input.h)");
}

void OnEvent(const SDL_Event& e) {
    switch (e.type) {
    case SDL_EVENT_GAMEPAD_ADDED:
        Open(e.gdevice.which);
        break;
    case SDL_EVENT_GAMEPAD_REMOVED:
        if (auto it = g_pads.find(e.gdevice.which); it != g_pads.end()) {
            Log("input: controller disconnected: %s", SDL_GetGamepadName(it->second));
            SDL_CloseGamepad(it->second);
            g_pads.erase(it);
        }
        break;
    case SDL_EVENT_KEY_DOWN: // (the function keys and the Capture button: Debug mode only, options.h Debug)
        if (!options::Debug()) break;
        if (e.key.scancode == SDL_SCANCODE_F9 && !e.key.repeat) g_capture_key = true;
        if (e.key.scancode == SDL_SCANCODE_F10 && !e.key.repeat) g_toggle_key = true;
        if (e.key.scancode == SDL_SCANCODE_F8 && !e.key.repeat) g_fps_key = true;
        if (e.key.scancode == SDL_SCANCODE_F7 && !e.key.repeat) g_flag_key = true;
        break;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        if (e.gbutton.button == SDL_GAMEPAD_BUTTON_MISC1 && options::Debug()) g_flag_key = true; // (not a Switch button)
        break;
    case SDL_EVENT_KEY_UP:
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_WINDOW_FOCUS_LOST:
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        break;
    default:
        return;
    }
    const Pad p = Poll();
    std::lock_guard lock(g_mu);
    // RT_TRACE_INPUT=1: each change of the buttons, on RT_HLE_TRACE's clock (presses against traced calls)
    static const bool trace = std::getenv("RT_TRACE_INPUT") && *std::getenv("RT_TRACE_INPUT") == '1';
    if (trace && p.buttons != g_pad.buttons) {
        Log("input: t=%.4f buttons 0x%llx", hle::TraceSeconds(), static_cast<unsigned long long>(p.buttons));
    }
    g_pad = p;
}

Pad Current() {
    std::lock_guard lock(g_mu);
    return g_pad;
}

bool TakeCaptureKey() { return g_capture_key.exchange(false); }
bool TakeToggleKey() { return g_toggle_key.exchange(false); }
bool TakeFpsKey() { return g_fps_key.exchange(false); }
bool TakeFlagKey() { return g_flag_key.exchange(false); }

} // namespace rt::input
