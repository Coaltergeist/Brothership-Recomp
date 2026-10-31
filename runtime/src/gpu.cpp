// The runtime's implementations of the shim interfaces video_core is built
// against (runtime/gpu/shim): Core::System, Core::Memory::Memory, the timing
// and perf-stat stand-ins, and a headless EmuWindow. See runtime/gpu/CMakeLists.txt.
#include "gpu.h"
#include "options.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "nvn/layer.h"
#include "npdm.h"
#include "rt.h"
#include "services.h"
#include "threads.h"
#include "window.h"

#include "common/logging.h"
#include "common/scratch_buffer.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/device_memory.h"
#include "core/gpu_dirty_memory_manager.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/frontend/graphics_context.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/rasterizer_download_area.h"
#include "video_core/renderer_base.h"
#include "video_core/video_core.h"

namespace {

struct State {
    Core::System system;
    Core::DeviceMemory device_memory;
    Core::Timing::CoreTiming timing;
    Core::PerfStats perf;
    Core::Memory::Memory memory;
    std::optional<Tegra::Host1x::Host1x> host1x;
    std::optional<Tegra::GPU> gpu;
    Core::GPUDirtyMemoryManager dirty; // CPU-written cached pages, until video_core gathers them
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    unsigned long long frames = 0;
};

State& S() {
    static State s;
    return s;
}

u64 Guest(Common::ProcessAddress a) {
    return GetInteger(a) & 0xFFFF'FFFF'FFFFULL;
}

} // namespace

namespace rt::gpu {

void Init() {
    auto& s = S();
    if (s.gpu) return;
    // drippu keeps its logs and caches in ./user when that directory exists
    // ("portable" mode); without it they would go to ~/.local/share.
    std::filesystem::create_directories("user");
    if (std::getenv("RT_GPU_LOG")) {
        Common::Log::Initialize(); // to user/log (written at exit) ...
        Common::Log::SetColorConsoleBackendEnabled(true); // ... and live to stdout
    }
    // With the NVN layer (RT_GPU=nvn) the layer renders and owns the window;
    // video_core only serves nvdrv, with its null renderer.
    const char* r = std::getenv("RT_GPU_RENDERER");
    const bool vulkan = r && std::string(r) == "vulkan" && !nvn::Enabled();
    Settings::values.renderer_backend.SetValue(vulkan ? Settings::RendererBackend::Vulkan
                                                      : Settings::RendererBackend::Null);
    // Asynchronous by default, as drippu's: the game runs ahead of the GPU thread
    // (fence callbacks then run on video_core's threads; nvdrv takes the global
    // lock for them). The title screen ran at 22.7 fps against 12.6 synchronous.
    // RT_GPU_SYNC=1 makes every submission wait for the GPU (for debugging).
    const bool async = std::getenv("RT_GPU_SYNC") == nullptr;
    Settings::values.use_asynchronous_gpu_emulation.SetValue(async);
    // In a window, show the newest frame without waiting for the monitor: the
    // game paces itself on our vsync event (vi), and a second wait here would
    // distort the frame-rate measurements. drippu falls back to FIFO if needed.
    const char* v = options::Get("RT_VSYNC");
    const std::string vsync = v ? v : "mailbox";
    Settings::values.vsync_mode.SetValue(vsync == "fifo"        ? Settings::VSyncMode::Fifo
                                         : vsync == "immediate" ? Settings::VSyncMode::Immediate
                                                                : Settings::VSyncMode::Mailbox);

    s.device_memory.buffer.base = g_mem.DeviceBacking();
    s.host1x.emplace(s.system);
    VideoCore::CreateGPU(s.gpu, window::Emu(), s.system);
    if (!s.gpu) Fatal("video_core: GPU initialisation failed");
    // The GPU thread (as drippu's frontends start it). Even "synchronous" mode
    // runs command lists there; the submitting thread waits for them.
    s.gpu->Start();
    Log("gpu  video_core up: %s renderer, %s", !vulkan ? "null" : window::IsOpen() ? "Vulkan, windowed" : "headless Vulkan",
        async ? "asynchronous" : "synchronous");
}

Core::System& System() {
    return S().system;
}

void WaitForFences(const std::vector<DisplayFence>& fences) {
    auto& syncpoints = S().host1x->GetSyncpointManager();
    for (const auto& f : fences) {
        if (f.id >= 0) syncpoints.WaitGuest(static_cast<u32>(f.id), f.value);
    }
}

bool RendererDraws() {
    return Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
}

u32 SyncpointGuestValue(u32 id) {
    return S().host1x->GetSyncpointManager().GetGuestSyncpointValue(id);
}

u32 SyncpointHostValue(u32 id) {
    return S().host1x->GetSyncpointManager().GetHostSyncpointValue(id);
}

unsigned long long ProgramId() {
    return TitleNpdm().program_id;
}

void OnCpuWrite(u64 va, u64 size) {
    auto& s = S();
    if (!s.gpu) return;
    // drippu keeps these per CPU core; per host thread is the same here.
    thread_local Common::ScratchBuffer<u32> scratch;
    thread_local DAddr last_page = ~DAddr{0}; // drippu: rasterizer_write_areas
    auto& device_memory = s.system.Host1x().MemoryManager();
    for (u64 at = va, end = va + size; at < end;) {
        const u64 n = std::min((at | Core::Memory::YUZU_PAGEMASK) + 1, end) - at;
        if (const u8* p = g_mem.DevicePointer(at)) {
            device_memory.ApplyOpOnPointer(p, scratch, [&](DAddr address) {
                // OnCPUWrite either invalidates now (false) or asks for the
                // page to be collected for the next gather (true).
                if (last_page != address >> Core::Memory::YUZU_PAGEBITS) {
                    if (!s.gpu->OnCPUWrite(address, n)) return;
                    last_page = address >> Core::Memory::YUZU_PAGEBITS;
                }
                // drippu keeps one collector per CPU core; ours is shared by the
                // guest threads, and Collect is not safe to call concurrently.
                static std::mutex collect_mutex;
                std::lock_guard lock(collect_mutex);
                s.dirty.Collect(address, n);
            });
        }
        at += n;
    }
}

void OnCpuRead(u64 va, u64 size) {
    auto& s = S();
    if (!s.gpu) return;
    thread_local Common::ScratchBuffer<u32> scratch;
    thread_local VideoCore::RasterizerDownloadArea area{}; // drippu: rasterizer_read_areas
    auto& device_memory = s.system.Host1x().MemoryManager();
    for (u64 at = va, end = va + size; at < end;) {
        const u64 n = std::min((at | Core::Memory::YUZU_PAGEMASK) + 1, end) - at;
        if (const u8* p = g_mem.DevicePointer(at)) {
            device_memory.ApplyOpOnPointer(p, scratch, [&](DAddr address) {
                if (area.start_address <= address && address + n <= area.end_address) return;
                // A flush waits for the GPU thread, which may need the global lock
                // (fence callbacks in nvdrv), so let go of it meanwhile.
                RunWithoutGil([&] { area = s.gpu->OnCPURead(address, n); });
            });
        }
        at += n;
    }
}

bool Composite(const std::vector<DisplayLayer>& in_layers, const std::vector<DisplayFence>& fences,
               const std::string& capture_path) {
    auto& s = S();
    std::vector<Tegra::FramebufferConfig> layers;
    for (const auto& layer : in_layers) {
        layers.push_back(Tegra::FramebufferConfig{
            .address = layer.device_address,
            .offset = layer.offset,
            .width = layer.width,
            .height = layer.height,
            .stride = layer.stride,
            .pixel_format = static_cast<::Service::android::PixelFormat>(layer.format),
            .transform_flags = static_cast<::Service::android::BufferTransformFlags>(layer.transform),
            .crop_rect = {layer.crop_left, layer.crop_top, layer.crop_right, layer.crop_bottom},
            .blending = layer.premultiplied ? Tegra::BlendMode::Premultiplied : Tegra::BlendMode::Opaque,
        });
    }
    std::vector<::Service::Nvidia::NvFence> nv_fences;
    for (const auto& f : fences) {
        if (f.id >= 0) nv_fences.push_back({f.id, f.value});
    }

    // The screenshot is taken during the composite, after the fences have passed.
    bool capturing = false;
    auto& renderer = s.gpu->Renderer();
    if (!capture_path.empty() && Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan &&
        !renderer.IsScreenshotPending()) {
        // The game's resolution, whatever size the window has.
        const auto out_layout = Layout::DefaultFrameLayout(1280, 720);
        auto bits = std::make_shared<std::vector<u8>>(static_cast<size_t>(out_layout.width) * out_layout.height * 4);
        renderer.RequestScreenshot(
            bits->data(),
            [bits, out_layout, capture_path](bool) { // on the renderer's thread: encode on the capture thread
                CaptureAsync([bits, out_layout, capture_path] {
                    if (!WriteBgraPng(capture_path, out_layout.width, out_layout.height, bits->data())) {
                        Log("gpu  could not write %s", capture_path.c_str());
                    }
                });
            },
            out_layout);
        capturing = true;
    }
    // RequestComposite waits for the GPU thread, which may need the global lock
    // first (nvdrv fence callbacks), so let go of it meanwhile.
    RunWithoutGil([&] { s.gpu->RequestComposite(std::move(layers), std::move(nv_fences)); });
    return capturing;
}

} // namespace rt::gpu

// --- Shim implementations ----------------------------------------------------

namespace Core {

System& System::GetInstance() { return S().system; }
bool System::IsPoweredOn() const { return true; }
// No per-core dirty-memory managers: CPU writes are not tracked (see Memory below).
void System::GatherGPUDirtyMemory(std::function<void(PAddr, size_t)>& callback) {
    S().dirty.Gather(callback);
}
Memory::Memory& System::ApplicationMemory() { return S().memory; }
Tegra::GPU& System::GPU() { return *S().gpu; }
const Tegra::GPU& System::GPU() const { return *S().gpu; }
Tegra::Host1x::Host1x& System::Host1x() { return *S().host1x; }
const Tegra::Host1x::Host1x& System::Host1x() const { return *S().host1x; }
Core::DeviceMemory& System::DeviceMemory() { return S().device_memory; }
const Core::DeviceMemory& System::DeviceMemory() const { return S().device_memory; }
Timing::CoreTiming& System::CoreTiming() { return S().timing; }
const Timing::CoreTiming& System::CoreTiming() const { return S().timing; }
PerfStats& System::GetPerfStats() { return S().perf; }
void System::RegisterHostThread() {}

u64 Timing::CoreTiming::GetGPUTicks() const {
    // GM20B timer: 614.4 MHz = 384/625 ticks per ns.
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - S().start).count();
    return static_cast<u64>(ns) / 625 * 384;
}

void PerfStats::EndGameFrame() {
    const auto n = ++S().frames;
    if (n <= 5 || n % 60 == 0) rt::Log("gpu  frame %llu", n);
}

namespace Memory {

// The device memory manager turns this pointer into a physical address, so it
// must point into the device backing: only device-visible pages have one.
u8* Memory::GetPointerSilent(Common::ProcessAddress vaddr) {
    return rt::g_mem.DevicePointer(Guest(vaddr));
}

// Lifted code's stores and the runtime's writes are checked (Memory::Access),
// so writes to these pages reach rt::gpu::OnCpuWrite. Only Raw() writes and
// guest-side DMA bypass it.
void Memory::RasterizerMarkRegionCached(Common::ProcessAddress vaddr, u64 size, bool cached) {
    rt::g_mem.MarkGpuCached(Guest(vaddr), size, cached);
}

bool Memory::ReadBlock(Common::ProcessAddress src, void* dest, std::size_t size) {
    std::memcpy(dest, rt::g_mem.Raw(Guest(src)), size);
    return true;
}
bool Memory::ReadBlockUnsafe(Common::ProcessAddress src, void* dest, std::size_t size) {
    return ReadBlock(src, dest, size);
}
bool Memory::WriteBlock(Common::ProcessAddress dest, const void* src, std::size_t size) {
    std::memcpy(rt::g_mem.Raw(Guest(dest)), src, size);
    return true;
}
bool Memory::WriteBlockUnsafe(Common::ProcessAddress dest, const void* src, std::size_t size) {
    return WriteBlock(dest, src, size);
}
bool Memory::WriteBlockCached(Common::ProcessAddress dest, const void* src, std::size_t size) {
    return WriteBlock(dest, src, size);
}
const u8* Memory::GetSpan(VAddr src, std::size_t) const {
    return rt::g_mem.Raw(src & 0xFFFF'FFFF'FFFFULL);
}
u8* Memory::GetSpan(VAddr src, std::size_t) {
    return rt::g_mem.Raw(src & 0xFFFF'FFFF'FFFFULL);
}

} // namespace Memory
} // namespace Core
