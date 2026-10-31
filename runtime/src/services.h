// Hooks between HLE services that live in different files.
#pragma once

#include <functional>
#include <string>

#include "rt.h"

namespace rt {

/// The one local user account (acc:u0), also preselected at launch (am).
struct AccountUid {
    u64 lo, hi;
};
constexpr AccountUid kAccountUser{0x0000'0000'0000'0001, 0x5265'636F'6D70'0001}; // any non-zero id

/// RT_DOCKED=1 (RESOLUTION_PLAN Phase 0, a test): am, apm and vi report docked mode (operation and performance
/// mode 1, a 1920x1080 display) instead of handheld; the controller stays the handheld one (hid.cpp).
bool ReportDocked();

/// Called once per presented frame (vi/binder QueueBuffer): refreshes input.
void HidOnFrame(u64 frame);

/// A game event the input script can wait for (RT_INPUT "@name"): today the
/// first RomFS read of each file, named by its path.
void InputEvent(const std::string& name);

/// Guest memory behind an nvmap id (nvdrv), for reading presented buffers.
bool NvMapLookup(u32 id, u64& address, u64& size);
/// Pins an nvmap id into the SMMU and returns its device address (0 if the id
/// is unknown or unallocated), as drippu's GraphicBuffer does for display buffers.
u64 NvMapPin(u32 id);
void NvMapUnpin(u32 id);
/// Makes runtime-owned guest memory [va, va + size) device-visible, maps it into
/// the SMMU for good and returns its device address (the overlay surface).
u64 PinHostRegion(u64 va, u64 size);

/// A presented buffer, described the way the guest configured it.
struct PresentedBuffer {
    u64 address;        // guest address of the slot's pixels
    u32 width, height;
    u32 pitch;          // bytes per row of the block-linear surface
    u32 block_height_log2;
    u32 format;         // Android pixel format: 1 RGBA8888, 4 RGB565, ...
};

/// Copies a block-linear buffer out of guest memory now, and deswizzles it and writes it as a PNG on the
/// capture thread (failures are logged there).
void CapturePng(const PresentedBuffer& buf, const std::string& path);
/// Writes linear BGRA8 pixels (as the renderer's screenshot returns them) as a PNG, on the calling thread.
bool WriteBgraPng(const std::string& path, u32 width, u32 height, const u8* bgra);
/// Runs `job` on one of the capture threads (capture.cpp; jobs may finish in any order). A capture is
/// dropped (and logged) while 48 are still waiting.
void CaptureAsync(std::function<void()> job);
/// Waits (up to 30 s) until every queued capture is written: before the runtime exits.
void FinishCaptures();

} // namespace rt
