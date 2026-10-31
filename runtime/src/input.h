// Host input for the guest's controller (hid.cpp): the keyboard and game controllers through SDL3, read on
// the main thread (the window's event loop) and taken by hid once per presented frame.
//
// Controllers: any SDL gamepad, buttons by position as on a Switch (right face button A, bottom B, top X,
// left Y; an Xbox pad's B is A), shoulders L/R, triggers ZL/ZR, Start/Back Plus/Minus, sticks and their clicks.
// Keyboard: WASD left stick, IJKL right stick, arrows the d-pad, Z/X/C/V A/B/X/Y, E/U L/R, Q/O ZL/ZR,
// Enter/Backspace Plus/Minus, F/N the stick clicks. Sticks: a 10 % radial dead zone; stick directions also
// set the stick-direction button bits (hid.cpp). gamecontrollerdb.txt in the working directory adds SDL mappings
// for unknown controllers. RT_HEADLESS has no window and no host input. Compared with LASRecomp's
// runtime/hle/input.c (beta_5, 2026-10-19): same pad layout; its keyboard uses K/J/L/I for A/B/X/Y and the numpad
// for the right stick, and it reports docked mode with a Pro Controller (we report handheld).
#pragma once

#include "rt.h"

typedef union SDL_Event SDL_Event;

namespace rt::input {

/// The controller as the host's devices hold it now.
struct Pad {
    u64 buttons = 0;           // HidNpadButton bits (A 1 << 0 ... Down 1 << 15)
    s32 lx = 0, ly = 0;        // left stick, -0x7FFF..0x7FFF, up positive (as hid reports it)
    s32 rx = 0, ry = 0;        // right stick
};

/// Main thread, after SDL_Init: starts the gamepad subsystem (controllers connected now arrive as events).
void Init();

/// Main thread: every SDL event (only keyboard and gamepad ones are used).
void OnEvent(const SDL_Event& e);

/// Any thread: the latest state (all zero without a window).
Pad Current();

/// Any thread: whether F9 (the layer's debug capture, nvn/window.cpp) was pressed since the last call.
bool TakeCaptureKey();
/// Any thread: whether F10 (the layer's A/B toggle, nvn/window.cpp) was pressed since the last call.
bool TakeToggleKey();
/// Any thread: whether F8 (the frame-rate cycle, nvn/window.cpp) was pressed since the last call.
bool TakeFpsKey();
/// Any thread: whether F7 or a controller's Capture/Share button (the player's flag, nvn/window.cpp) was
/// pressed since the last call.
bool TakeFlagKey();

} // namespace rt::input
