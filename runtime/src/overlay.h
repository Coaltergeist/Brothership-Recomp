// On-screen overlay: the last second's frame rate and frame times (perf.h),
// drawn by the CPU with a built-in 5x7 font into runtime-owned guest memory and
// composited by video_core as a second, premultiplied-alpha layer over the game.
//
// RT_OVERLAY=1/0 turns it on or off; the default is on with a window and off
// headless (so captures and tests see only the game).
//
// The surface lives in a runtime region of the guest address space
// (kOverlayBase, inaccessible to the guest), made device-visible and mapped
// into the SMMU like a game buffer. The renderer reads it from memory on every
// composite (no game texture covers it), so a CPU redraw needs no invalidation.
#pragma once

#include <optional>
#include <vector>

#include "gpu.h"

namespace rt::overlay {

/// The overlay layer to composite with this frame, or nothing when it is off.
/// Called once per presented frame (vi), with the kernel lock held.
std::optional<gpu::DisplayLayer> Layer();

/// The overlay's text box for a renderer that draws it itself (the NVN layer's window): its place in a
/// 1280x720 frame and its pixels (RGBA, premultiplied alpha, w x h), redrawn when a new second has been
/// measured; nothing when the overlay is off.
struct Box {
    u32 x, y, w, h;
    std::vector<u8> rgba;
};
std::optional<Box> CurrentBox();

/// Any thread: shows "FLAG <n>" next to the frame rate for a few seconds (the player's flag, nvn/window.cpp).
void ShowFlag(unsigned n);

} // namespace rt::overlay
