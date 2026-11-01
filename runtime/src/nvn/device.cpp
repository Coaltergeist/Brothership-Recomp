// NVNdevice and its builder, and the layer's Vulkan device.
#include "heavy.h"
#include "render.h"
#include "vk.h"

#include "../window.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>

#include "common/settings.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_surface.h"

namespace rt::nvn {

VulkanCore& Vk() {
    static VulkanCore* core = [] {
        auto* v = new VulkanCore; // lives until the process ends
        try {
            v->library = Vulkan::OpenLibrary();
            const bool windowed = window::IsOpen();
            const auto info = window::Emu().GetWindowInfo();
            v->instance = Vulkan::CreateInstance(*v->library, v->dld, VK_API_VERSION_1_1,
                                                 windowed ? info.type : Core::Frontend::WindowSystemType::Headless,
                                                 std::getenv("RT_VK_VALIDATION") != nullptr);
            if (windowed) v->surface = Vulkan::CreateSurface(v->instance, info);
            const auto devices = v->instance.EnumeratePhysicalDevices();
            const char* pick = std::getenv("RT_VK_DEVICE");
            size_t index = pick ? std::strtoul(pick, nullptr, 10) : 0;
            if (!pick) { // a discrete GPU first, then an integrated one, then anything (lavapipe in WSL)
                auto rank = [&](VkPhysicalDevice d) {
                    switch (Vulkan::vk::PhysicalDevice(d, v->dld).GetProperties().deviceType) {
                    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 0;
                    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
                    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
                    default: return 3;
                    }
                };
                for (size_t i = 1; i < devices.size(); ++i) {
                    if (rank(devices[i]) < rank(devices[index])) index = i;
                }
            }
            if (index >= devices.size()) Fatal("nvn: no Vulkan device %zu (%zu found)", index, devices.size());
            const Vulkan::vk::PhysicalDevice physical(devices[index], v->dld);
            // drippu enables VK_EXT_graphics_pipeline_library only when asked (off by default there, for its own
            // pipeline cache's use of it, which the layer does not have): the layer links its graphics pipelines
            // from libraries (pipeline.cpp GplOn; RT_NVN_GPL=0: whole pipelines, the extension left off).
            const char* gpl = std::getenv("RT_NVN_GPL");
            if (!(gpl && std::strcmp(gpl, "0") == 0) || std::getenv("RT_NVN_SHADERTEST_GPL")) {
                Settings::values.use_graphics_pipeline_library.SetValue(true);
            }
            v->device = std::make_unique<Vulkan::Device>(*v->instance, physical, *v->surface, v->dld);
            v->allocator = std::make_unique<Vulkan::MemoryAllocator>(*v->device);
            if (HeavyOn()) InstallDriverTimers(const_cast<Vulkan::vk::DeviceDispatch&>(v->device->GetDispatchLoader()));
        } catch (const std::exception& e) {
            Fatal("nvn: Vulkan initialisation failed: %s", e.what());
        }
        Log("nvn: Vulkan device %s (%s), %s", std::string(v->device->GetModelName()).c_str(),
            v->device->GetDriverName().c_str(), v->surface ? "presenting to the window" : "headless");
        StartPipelineCacheLoad(); // its thread uses Vk() once this initialisation has returned
        return v;
    }();
    return *core;
}

std::mutex& QueueLock() {
    static std::mutex m;
    return m;
}

namespace {

// --- Device ----------------------------------------------------------------------------------

struct DeviceBuilder { // inline in NVNdeviceBuilder
    u32 flags;
};

struct Device : Object {
    static constexpr Type kType = Type::kDevice;
    u32 flags = 0;
    u32 depth_mode = 0;    // NVN_DEPTH_MODE_NEAR_IS_MINUS_W
    u32 window_origin = 0; // NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT
    u64 debug_callback = 0, debug_data = 0;
};

std::atomic<s32> g_debug_domains{0};

// nvnDeviceGetInteger ids whose answer comes from the device, not the SDK: in
// SDK 12.3.2 these cases read the NVNdevice (0x56c090's jump table), and the
// values are what Brothership's driver answers on the console's GPU (the tap's
// device-info probe, 2026-10-15). Every other id is the game's SDK's own answer
// (API version, limits, alignments), asked with a zeroed stand-in device.
// Another SDK may read the device for other ids: the tap's shadow check
// (nvn shadow: ... DIFFERENT) shows them.
const std::map<u32, s32> kDeviceValues = {
    {23, 0x2},    {24, 0x100},  {43, 0x40000},    {45, 0x800},  {55, 0x80},   {56, 0x20},   {58, 0x10000},
    {59, 0x1},    {61, 0x4000}, {62, 0x8000000},  {63, 0x800},  {64, 0x4000}, {65, 0x4000}, {98, 0x20},
};

} // namespace

NVN_PROC(nvnDeviceBuilderSetDefaults, void, u64 builder) { Store(builder, DeviceBuilder{}); }
NVN_PROC(nvnDeviceBuilderSetFlags, void, u64 builder, u32 flags) {
    auto b = Load<DeviceBuilder>(builder);
    b.flags = flags;
    Store(builder, b);
}
NVN_PROC(nvnDeviceBuilderGetFlags, u32, u64 builder) { return Load<DeviceBuilder>(builder).flags; }

NVN_PROC(nvnDeviceInitialize, bool, u64 device, u64 builder) {
    Vk();
    Device* d = Create<Device>(device);
    d->flags = Load<DeviceBuilder>(builder).flags;
    Log("nvn: device initialized (flags 0x%x)", d->flags);
    return true;
}
NVN_PROC(nvnDeviceFinalize, void, u64 device) { Destroy<Device>(device); }
NVN_PROC(nvnDeviceSetDebugLabel, void, u64 device, u64 label) {
    Get<Device>(device);
    if (label) Log("nvn: device label \"%s\"", hle::ReadGuestString(label, 256).c_str());
}
NVN_PROC(nvnDeviceGetProcAddress, u64, u64 device, u64 name) {
    return name ? Lookup(hle::ReadGuestString(name, 256)) : 0;
}
s32 DeviceInteger(GuestContext* c, u32 which) {
    const auto it = kDeviceValues.find(which);
    return it != kDeviceValues.end() ? it->second : SdkDeviceInteger(c, which);
}

NVN_PROC(nvnDeviceGetInteger, void, u64 device, u32 which, u64 out) {
    hle::WriteGuest<s32>(out, DeviceInteger(c, which));
}
NVN_PROC(nvnDeviceSetDepthMode, void, u64 device, u32 mode) {
    Log("nvn: nvnDeviceSetDepthMode(%u)", mode);
    Get<Device>(device)->depth_mode = mode;
}
NVN_PROC(nvnDeviceGetDepthMode, u32, u64 device) { return Get<Device>(device)->depth_mode; }
NVN_PROC(nvnDeviceSetWindowOriginMode, void, u64 device, u32 mode) {
    Log("nvn: nvnDeviceSetWindowOriginMode(%u)", mode);
    Get<Device>(device)->window_origin = mode;
}
NVN_PROC(nvnDeviceGetWindowOriginMode, u32, u64 device) { return Get<Device>(device)->window_origin; }
NVN_PROC(nvnDeviceInstallDebugCallback, void, u64 device, u64 callback, u64 data, bool enable) {
    Device* d = Get<Device>(device);
    d->debug_callback = enable ? callback : 0;
    d->debug_data = data;
}
NVN_PROC(nvnDeviceGenerateDebugDomainId, s32, u64 device, u64 name) { return ++g_debug_domains; }
// Removed after SDK 8.2: empty there.
NVN_PROC(nvnDeviceApplyDeferredFinalizes, void, u64 device, s32 age) {}
NVN_PROC(nvnDeviceIsExternalDebuggerAttached, bool, u64 device) { return false; }
NVN_PROC(nvnDeviceGetCurrentTimestampInNanoseconds, u64, u64 device) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

// Fast clears: NVN registers clear values in the GPU's fast-clear table (a
// hardware speed-up with no visible effect) and answers whether the table had
// room. Vulkan needs no table: always room (Brothership registers 10 colours
// and 2 depths).
NVN_PROC(nvnDeviceRegisterFastClearColor, bool, u64 device, u64 color, u32 format) {
    Get<Device>(device);
    return true;
}
NVN_PROC(nvnDeviceRegisterFastClearColori, bool, u64 device, u64 color, u32 format) {
    Get<Device>(device);
    return true;
}
NVN_PROC(nvnDeviceRegisterFastClearColorui, bool, u64 device, u64 color, u32 format) {
    Get<Device>(device);
    return true;
}
NVN_PROC(nvnDeviceRegisterFastClearDepth, bool, u64 device, float depth) {
    Get<Device>(device);
    return true;
}

// Handles: the game's own SDK computes them from the ids alone (SDK 3.5.0,
// 8.2.0 and 12.3.2 checked; 3.5.0 differs in the sign extension of sampler
// handles), so its answer is NVN's encoding for that game; shaders decode it.
NVN_PROC(nvnDeviceGetTextureHandle, u64, u64 device, s32 texture, s32 sampler) {
    return SdkPure(c, "nvnDeviceGetTextureHandle", static_cast<u32>(texture), static_cast<u32>(sampler));
}
NVN_PROC(nvnDeviceGetSeparateTextureHandle, u64, u64 device, s32 texture) {
    return SdkPure(c, "nvnDeviceGetSeparateTextureHandle", static_cast<u32>(texture));
}
NVN_PROC(nvnDeviceGetSeparateSamplerHandle, u64, u64 device, s32 sampler) {
    return SdkPure(c, "nvnDeviceGetSeparateSamplerHandle", static_cast<u32>(sampler));
}
NVN_PROC(nvnDeviceGetImageHandle, u64, u64 device, s32 image) {
    return SdkPure(c, "nvnDeviceGetImageHandle", static_cast<u32>(image));
}
NVN_PROC(nvnDeviceGetTexelFetchHandle, u64, u64 device, s32 texture) {
    return SdkPure(c, "nvnDeviceGetTexelFetchHandle", static_cast<u32>(texture));
}

ClipModes DeviceClipModes(u64 device) {
    const Device* d = Get<Device>(device);
    return {d->depth_mode, d->window_origin};
}

} // namespace rt::nvn
