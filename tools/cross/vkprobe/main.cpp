// Lists the Vulkan devices Windows offers (through vulkan-1.dll, loaded at run time).
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <windows.h>

#include <cstdio>
#include <vector>

int main() {
    std::printf("vkprobe: built %s %s with clang %s\n", __DATE__, __TIME__, __clang_version__);
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) { std::printf("no vulkan-1.dll (error %lu)\n", GetLastError()); return 1; }
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(lib, "vkGetInstanceProcAddr"));
    auto create = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "vkprobe", 1, nullptr, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    VkInstance inst;
    if (VkResult r = create(&ci, nullptr, &inst); r != VK_SUCCESS) { std::printf("vkCreateInstance: %d\n", r); return 1; }
    auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(inst, "vkEnumeratePhysicalDevices"));
    auto props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gipa(inst, "vkGetPhysicalDeviceProperties"));
    uint32_t n = 0;
    enumerate(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    enumerate(inst, &n, devs.data());
    for (VkPhysicalDevice d : devs) {
        VkPhysicalDeviceProperties p;
        props(d, &p);
        std::printf("device: %s, Vulkan %u.%u.%u, type %d\n", p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
                    VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion), p.deviceType);
    }
    return 0;
}
