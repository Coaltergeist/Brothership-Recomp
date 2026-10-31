// The runtime's side of video_core (drippu's GPU emulation, runtime/gpu):
// it implements the shim interfaces of runtime/gpu/shim and owns the GPU.
#pragma once

#include <string>
#include <vector>

#include "rt.h"

namespace Core {
class System;
}

namespace rt::gpu {

/// One presented buffer, from vi's GraphicBuffer (transaction 14) and
/// QueueBufferInput (transaction 7); the fields of drippu's HwcLayer.
struct DisplayLayer {
    u64 device_address; // nvmap handle pinned into the SMMU (NvMapPin)
    u32 offset, width, height;
    u32 stride; // in pixels
    u32 format; // Android pixel format
    u32 transform;
    s32 crop_left, crop_top, crop_right, crop_bottom;
    bool premultiplied = false; // blend over the layers below (the overlay); else opaque
};

/// A syncpoint the frame's rendering signals (NvFence).
struct DisplayFence {
    s32 id;
    u32 value;
};

/// A checked CPU write to GPU-cached guest memory (Memory::Access): tells
/// video_core, as drippu's Memory::HandleRasterizerWrite does. Written pages
/// are handed back when video_core gathers dirty memory.
void OnCpuWrite(u64 va, u64 size);

/// A checked CPU read of GPU-cached guest memory: first flushes what the GPU
/// wrote there (query results, semaphores, render-to-buffer) back to guest
/// memory, as drippu's Memory::HandleRasterizerDownload does.
void OnCpuRead(u64 va, u64 size);

/// Composites a frame (layers bottom to top) once its fences have passed, as
/// drippu's nvdisp_disp0::Composite does. When capture_path is set and the
/// renderer draws (Vulkan), the composited image is also written there as a
/// 1280x720 PNG and true is returned; the null renderer draws nothing and
/// returns false.
bool Composite(const std::vector<DisplayLayer>& layers, const std::vector<DisplayFence>& fences,
               const std::string& capture_path);

/// Creates Host1x and the GPU (once; nvdrv calls it when the first device opens).
/// RT_GPU_RENDERER=vulkan selects Vulkan, presenting to the window (window.h)
/// or headless; the default is the null renderer. RT_VSYNC=mailbox (default),
/// fifo or immediate picks the window's present mode.
void Init();
Core::System& System();
/// Blocks until every fence has passed (as the guest sees the syncpoints).
/// Call without the global lock: the GPU thread may need it to get there.
void WaitForFences(const std::vector<DisplayFence>& fences);
/// Whether the renderer draws (Vulkan), so captures come from its output.
bool RendererDraws();
/// A syncpoint's value as the guest sees it and as the host GPU has reached it (for traces).
u32 SyncpointGuestValue(u32 id);
u32 SyncpointHostValue(u32 id);
/// The title's program id (video_core keys per-game data on it), from the NPDM (npdm.h).
unsigned long long ProgramId();

} // namespace rt::gpu
