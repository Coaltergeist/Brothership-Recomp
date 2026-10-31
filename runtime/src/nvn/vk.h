// The layer's Vulkan device (device.cpp), for the files that use Vulkan.
#pragma once

#include "nvn.h"

#include <memory>
#include <mutex>

#include "common/dynamic_library.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace rt::nvn {

// One instance and device for the layer, from drippu's vulkan_common (as its
// renderer makes them). It presents to the runtime's window when one is open
// (video_core then runs its null renderer, gpu.cpp). RT_VK_DEVICE=<n> picks a
// physical device (default: the first discrete GPU, else integrated, else any),
// RT_VK_VALIDATION=1 the validation layers.
struct VulkanCore {
    std::shared_ptr<Common::DynamicLibrary> library;
    Vulkan::vk::InstanceDispatch dld;
    Vulkan::vk::Instance instance;
    Vulkan::vk::SurfaceKHR surface;
    std::unique_ptr<Vulkan::Device> device;
    std::unique_ptr<Vulkan::MemoryAllocator> allocator;
};

/// Created at the first use (nvnDeviceInitialize).
VulkanCore& Vk();

/// Held around every use of the device's one queue (vkQueueSubmit, vkQueuePresentKHR, vkDeviceWaitIdle):
/// Vulkan wants those externally synchronised, and the NVN queues and the window submit from different threads.
std::mutex& QueueLock();

} // namespace rt::nvn
