// The host window (SDL3) that video_core presents to, or a headless stand-in.
//
// The process's main thread owns the window and runs its event loop (SDL and
// the OSes want that); the guest's main thread runs on a thread of its own
// (rt.cpp). RT_HEADLESS=1 opens no window: video_core then renders off-screen
// and frames are only seen through captures, as before.
//
// The title shows the last second's frame rate (perf.h). Closing the window
// ends the process.
#pragma once

#include "rt.h"

namespace Core::Frontend {
class EmuWindow;
}
struct SDL_Window;

namespace rt::window {

/// Opens the window unless RT_HEADLESS=1. Call on the main thread, before guest
/// code starts. Returns whether a window is open.
bool Open();
bool IsOpen();

/// Sets the window mode and size again from the options and chooses the resolution again (the start screen may have
/// changed them). Before guest code starts.
void Reapply();

/// The SDL window (the start screen draws in it), or nullptr headless.
SDL_Window* Sdl();

/// The EmuWindow video_core draws to (the window, or the headless stand-in).
Core::Frontend::EmuWindow& Emu();

/// Runs the event loop on the main thread; returns only by ending the process.
[[noreturn]] void RunEventLoop();

} // namespace rt::window
