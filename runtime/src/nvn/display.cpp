// The NVN layer's image on the host window.
//
// A swapchain on the window's surface (the layer's Vulkan device is made for it, device.cpp). Each
// nvnQueuePresentTexture shows the presented texture: after a full barrier (the game's work on the
// same Vulkan queue comes earlier in submission order), the swapchain image is cleared black, the
// texture (with RT_AA=smaa its SMAA'd copy, smaa.h) blitted into the largest rectangle of its aspect ratio,
// the overlay's text box (overlay.h)
// copied over its top-left corner, and the image presented. The swapchain has the texture's sRGB-ness
// (Brothership's window textures are BGRA8 sRGB, 0x9c), so the bytes reach the screen as the game wrote
// them, as in the captures. Present mode from RT_VSYNC as on the driver path (on, the default: mailbox - the
// game paces itself on the window model, window.cpp; FIFO where mailbox is missing; off: immediate). Re-created when the
// window changes size or the driver says so.
//
// Slow presents: whenever a step of showing a frame (the display lock, a new swapchain, the wait for the
// frame slot's earlier copy, vkAcquireNextImageKHR, vkQueueSubmit + vkQueuePresentKHR) or the time since the
// previous frame shown takes over 50 ms, a "nvn: window: slow present" line with the uptime and all of them
// (always on; for freezes the game's own presents don't show, 2026-10-19).
//
// Not yet: the overlay box blended (it is copied, so its translucent black is opaque), scaling it with
// the window.
#include "display.h"

#include "images.h"
#include "smaa.h"
#include "vk.h"

#include "../overlay.h"
#include "../options.h"
#include "../threads.h"
#include "../window.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rt::nvn {

namespace {

constexpr u32 kFramesInFlight = 2;

struct Frame {
    Vulkan::vk::CommandPool pool;
    Vulkan::vk::CommandBuffers buffers;
    Vulkan::vk::Semaphore acquired;
    Vulkan::vk::Fence done;
    Vulkan::vk::Buffer overlay; // the text box, host-visible
    std::shared_ptr<const TextureImage> shown; // alive until `done`
};

struct Display {
    std::mutex mu;
    bool failed = false;
    Vulkan::vk::SwapchainKHR swapchain;
    std::vector<VkImage> images;
    std::vector<Vulkan::vk::Semaphore> rendered; // per swapchain image: signalled by the copy, waited by present
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    bool srgb = false, stale = false;
    std::array<Frame, kFramesInFlight> frames;
    u32 next = 0;
    // The start screen's (ShowUi): a render pass clearing the image, a view and framebuffer per swapchain image
    Vulkan::vk::RenderPass ui_pass;
    std::vector<Vulkan::vk::ImageView> ui_views;
    std::vector<Vulkan::vk::Framebuffer> ui_framebuffers;
};

Display& D() {
    static Display d;
    return d;
}

/// RT_VSYNC=on (default): mailbox, no tearing and no wait for the monitor (the game paces itself); off: immediate,
/// may tear. Each falls back to the next of mailbox / FIFO the driver has (FIFO is always there).
VkPresentModeKHR PresentMode(const std::vector<VkPresentModeKHR>& modes) {
    const char* v = options::Get("RT_VSYNC");
    const bool off = v && std::strcmp(v, "off") == 0;
    const auto has = [&](VkPresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
    if (off && has(VK_PRESENT_MODE_IMMEDIATE_KHR)) return VK_PRESENT_MODE_IMMEDIATE_KHR;
    if (has(VK_PRESENT_MODE_MAILBOX_KHR)) return VK_PRESENT_MODE_MAILBOX_KHR;
    return VK_PRESENT_MODE_FIFO_KHR;
}

/// (Re)creates the swapchain; false while the window has no area (minimised).
bool Create(Display& d, bool srgb) {
    const auto& dev = *Vk().device;
    const auto& physical = dev.GetPhysical();
    const VkSurfaceKHR surface = *Vk().surface;
    const VkSurfaceCapabilitiesKHR caps = physical.GetSurfaceCapabilitiesKHR(surface);
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFF) extent = {1280, 720}; // the surface takes the swapchain's size
    if (extent.width == 0 || extent.height == 0) return false;
    {
        std::lock_guard lock(QueueLock());
        dev.GetLogical().WaitIdle(); // nothing may still use the old swapchain's images
    }
    const auto formats = physical.GetSurfaceFormatsKHR(surface);
    const VkFormat wanted[] = {srgb ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM,
                               srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM};
    // sRGB colour space only: a compositor with colour management may list the same format in linear or HDR
    // spaces too, which would show the game's sRGB values brightened.
    VkSurfaceFormatKHR format = formats.at(0);
    for (VkFormat w : wanted) {
        bool found = false;
        for (const auto& f : formats) {
            if (f.format == w && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) format = f, found = true;
        }
        if (found) break;
    }
    u32 count = std::max<u32>(caps.minImageCount + 1, 3);
    if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
    const VkCompositeAlphaFlagBitsKHR alpha = caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                                  ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                                  : static_cast<VkCompositeAlphaFlagBitsKHR>(
                                                        caps.supportedCompositeAlpha & -caps.supportedCompositeAlpha);
    d.ui_framebuffers.clear(); // (they use the old images' views)
    d.ui_views.clear();
    Vulkan::vk::SwapchainKHR old = std::move(d.swapchain);
    d.swapchain = dev.GetLogical().CreateSwapchainKHR({
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = surface,
        .minImageCount = count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, // (the start screen draws)
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = caps.currentTransform,
        .compositeAlpha = alpha,
        .presentMode = PresentMode(physical.GetSurfacePresentModesKHR(surface)),
        .clipped = VK_TRUE,
        .oldSwapchain = *old,
    });
    d.images = d.swapchain.GetImages();
    d.rendered.clear();
    for (size_t i = 0; i < d.images.size(); ++i) d.rendered.push_back(dev.GetLogical().CreateSemaphore());
    d.format = format.format;
    d.extent = extent;
    d.srgb = srgb;
    d.stale = false;
    Log("nvn: window swapchain %ux%u, format %d, colour space %d, %zu images", extent.width, extent.height, format.format,
        format.colorSpace, d.images.size());
    return true;
}

void InitFrames(Display& d) {
    const auto& logical = Vk().device->GetLogical();
    for (Frame& f : d.frames) {
        f.pool = logical.CreateCommandPool({
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = Vk().device->GetGraphicsFamily(),
        });
        f.buffers = f.pool.Allocate(1);
        f.acquired = logical.CreateSemaphore();
        f.done = logical.CreateFence({
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        });
    }
}

void Barrier(const Vulkan::vk::CommandBuffer& cmd, VkImage image, VkImageLayout from, VkImageLayout to,
             VkPipelineStageFlags src_stage, VkAccessFlags src, VkAccessFlags dst) {
    cmd.PipelineBarrier(src_stage, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                        VkImageMemoryBarrier{
                            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            .pNext = nullptr,
                            .srcAccessMask = src,
                            .dstAccessMask = dst,
                            .oldLayout = from,
                            .newLayout = to,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = image,
                            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                        });
}

/// Copies the overlay's text box into the swapchain image at (x, y) of the picture.
void DrawOverlay(Display& d, Frame& f, const Vulkan::vk::CommandBuffer& cmd, VkImage image, s32 x0, s32 y0) {
    const auto box = overlay::CurrentBox();
    if (!box) return;
    const s32 x = x0 + static_cast<s32>(box->x), y = y0 + static_cast<s32>(box->y);
    if (x < 0 || y < 0 || x + box->w > d.extent.width || y + box->h > d.extent.height) return;
    const u64 size = u64{box->w} * box->h * 4;
    if (!f.overlay) {
        f.overlay = Vk().allocator->CreateBuffer(
            {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = size,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            Vulkan::MemoryUsage::Upload);
    }
    u8* out = f.overlay.Mapped().data();
    const bool bgr = d.format == VK_FORMAT_B8G8R8A8_SRGB || d.format == VK_FORMAT_B8G8R8A8_UNORM;
    for (u64 i = 0; i < size; i += 4) { // premultiplied over black = the colour channels
        out[i + 0] = box->rgba[i + (bgr ? 2 : 0)];
        out[i + 1] = box->rgba[i + 1];
        out[i + 2] = box->rgba[i + (bgr ? 0 : 2)];
        out[i + 3] = 0xFF;
    }
    f.overlay.Flush();
    const VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {x, y, 0},
        .imageExtent = {box->w, box->h, 1},
    };
    cmd.CopyBufferToImage(*f.overlay, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
}

} // namespace

namespace {
/// The steps of one ShowInWindow, in ms; logged when one of them is slow.
struct PresentTimes {
    using Clock = std::chrono::steady_clock;
    static constexpr double kSlowMs = 50;
    Clock::time_point at = Clock::now();
    double since_last = 0, lock = 0, create = 0, copy_wait = 0, acquire = 0, present = 0;
    /// The ms since the last call (or construction), for the step that just ended.
    double Lap() {
        const Clock::time_point now = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - at).count();
        at = now;
        return ms;
    }
    ~PresentTimes() {
        if (std::max({since_last, lock, create, copy_wait, acquire, present}) <= kSlowMs) return;
        Log("nvn: window: slow present (t=%.3f s): %.1f ms since the last one; lock %.1f, new swapchain %.1f, "
            "earlier copy %.1f, acquire %.1f, submit + present %.1f ms",
            Uptime(), since_last, lock, create, copy_wait, acquire, present);
    }
};
} // namespace

void ShowInWindow(const TextureState& t) {
    if (!Vk().surface) return;
    static PresentTimes::Clock::time_point last{};
    PresentTimes times;
    if (last != PresentTimes::Clock::time_point{}) {
        times.since_last = std::chrono::duration<double, std::milli>(times.at - last).count();
    }
    last = times.at; // the callers present one at a time (the game's presenting thread)
    Display& d = D();
    std::lock_guard lock(d.mu);
    times.lock = times.Lap();
    if (d.failed) return;
    const u32 f_id = t.settings.format;
    const bool srgb = f_id == 0x37 || f_id == 0x38 || f_id == 0x9b || f_id == 0x9c;
    if (!d.frames[0].pool) InitFrames(d);
    if (!d.swapchain || d.stale || d.srgb != srgb) {
        times.Lap();
        const bool made = Create(d, srgb);
        times.create = times.Lap();
        if (!made) return;
    }
    const u32 slot = d.next;
    Frame& f = d.frames[slot];
    d.next = (d.next + 1) % kFramesInFlight;
    times.Lap();
    RunHostWait(kHostWaitPresent, [&] { f.done.Wait(); });
    times.copy_wait = times.Lap();
    f.shown.reset();

    const auto& dev = *Vk().device;
    const auto& logical = dev.GetLogical();
    u32 index = 0;
    const VkResult acquired = logical.AcquireNextImageKHR(*d.swapchain, ~0ULL, *f.acquired, VK_NULL_HANDLE, &index);
    times.acquire = times.Lap();
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        d.stale = true; // dropped; the next present makes a new swapchain
        return;
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
        Log("nvn: window: vkAcquireNextImageKHR failed (VkResult %d); the window stays as it is", acquired);
        d.failed = true;
        return;
    }
    if (acquired == VK_SUBOPTIMAL_KHR) d.stale = true;
    f.done.Reset();

    const Vulkan::vk::CommandBuffer cmd(f.buffers[0], dev.GetDispatchLoader());
    cmd.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    const VkImage image = d.images[index];
    const auto texture = ImageFor(t, cmd);
    NoteImageUse(t, ImageUse::kPresent);
    f.shown = texture;
    // everything the game submitted before (earlier in submission order on the same queue)
    Barrier(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    FullBarrier(cmd);
    // RT_AA=smaa: the game's image anti-aliased at its own size first (smaa.h)
    const VkImage shown = SmaaOn() ? SmaaApply(cmd, slot, *texture, {texture->UpSize(t.settings.width),
                                                                      texture->UpSize(t.settings.height)})
                                   : *texture->image;
    const VkClearColorValue black{};
    const VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dev.GetDispatchLoader().vkCmdClearColorImage(*cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &whole);
    Barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    // the largest rectangle of the texture's aspect ratio, centred
    const u32 tw = t.settings.width, th = t.settings.height;
    u32 w = d.extent.width, h = static_cast<u32>(u64{d.extent.width} * th / tw);
    if (h > d.extent.height) h = d.extent.height, w = static_cast<u32>(u64{d.extent.height} * tw / th);
    const s32 x = static_cast<s32>((d.extent.width - w) / 2), y = static_cast<s32>((d.extent.height - h) / 2);
    const VkImageBlit blit{
        .srcSubresource = {texture->aspect, 0, 0, 1},
        .srcOffsets = {{0, 0, 0}, {static_cast<s32>(texture->UpSize(tw)), static_cast<s32>(texture->UpSize(th)), 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffsets = {{x, y, 0}, {x + static_cast<s32>(w), y + static_cast<s32>(h), 1}},
    };
    cmd.BlitImage(shown, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, blit,
                  VK_FILTER_LINEAR);
    Barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    DrawOverlay(d, f, cmd, image, x, y);
    Barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    cmd.End();

    const VkSemaphore wait = *f.acquired, signal = *d.rendered[index];
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkSubmitInfo submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait,
        .pWaitDstStageMask = &stage,
        .commandBufferCount = 1,
        .pCommandBuffers = cmd.address(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal,
    };
    const VkSwapchainKHR swapchain = *d.swapchain;
    const VkPresentInfoKHR present{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &signal,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &index,
        .pResults = nullptr,
    };
    VkResult presented;
    {
        times.Lap();
        std::lock_guard queue_lock(QueueLock());
        if (dev.GetGraphicsQueue().Submit(submit, *f.done) != VK_SUCCESS) Fatal("nvn: window: vkQueueSubmit failed");
        presented = dev.GetPresentQueue().Present(present);
        times.present = times.Lap();
    }
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) d.stale = true;
    else if (presented != VK_SUCCESS) Log("nvn: window: vkQueuePresentKHR: VkResult %d", presented);
}


namespace {

/// The start screen's swapchain: plain (UNORM) colour, as ImGui's colours are meant, and its render pass and
/// framebuffers. False while the window has no area.
bool PrepareUi(Display& d) {
    if (!d.frames[0].pool) InitFrames(d);
    if (!d.swapchain || d.stale || d.srgb) {
        if (!Create(d, false)) return false;
    }
    const auto& logical = Vk().device->GetLogical();
    if (!d.ui_pass) {
        const VkAttachmentDescription color{
            .flags = 0,
            .format = d.format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };
        const VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        const VkSubpassDescription subpass{
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachments = &ref,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = nullptr,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };
        // the image is acquired (the semaphore waits at this stage) before the clear writes it
        const VkSubpassDependency acquire{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = 0,
        };
        d.ui_pass = logical.CreateRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .attachmentCount = 1,
            .pAttachments = &color,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &acquire,
        });
    }
    if (d.ui_framebuffers.size() != d.images.size()) {
        d.ui_framebuffers.clear();
        d.ui_views.clear();
        for (VkImage image : d.images) {
            d.ui_views.push_back(logical.CreateImageView({
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = d.format,
                .components = {},
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            }));
            const VkImageView view = *d.ui_views.back();
            d.ui_framebuffers.push_back(logical.CreateFramebuffer({
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderPass = *d.ui_pass,
                .attachmentCount = 1,
                .pAttachments = &view,
                .width = d.extent.width,
                .height = d.extent.height,
                .layers = 1,
            }));
        }
    }
    return true;
}

} // namespace

bool UiTarget(VkRenderPass* pass, u32* images, VkFormat* format) {
    if (!Vk().surface) return false;
    Display& d = D();
    std::lock_guard lock(d.mu);
    if (!PrepareUi(d)) return false;
    *pass = *d.ui_pass;
    *images = static_cast<u32>(d.images.size());
    *format = d.format;
    return true;
}

bool ShowUi(const std::function<void(VkCommandBuffer)>& draw, VkExtent2D* extent) {
    if (!Vk().surface) return false;
    Display& d = D();
    std::lock_guard lock(d.mu);
    if (d.failed) return false;
    if (!PrepareUi(d)) return false;
    *extent = d.extent;
    Frame& f = d.frames[d.next];
    d.next = (d.next + 1) % kFramesInFlight;
    f.done.Wait();
    f.shown.reset();
    const auto& dev = *Vk().device;
    const auto& logical = dev.GetLogical();
    u32 index = 0;
    const VkResult acquired = logical.AcquireNextImageKHR(*d.swapchain, ~0ULL, *f.acquired, VK_NULL_HANDLE, &index);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        d.stale = true;
        return false;
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
        Log("nvn: window: vkAcquireNextImageKHR failed (VkResult %d) on the start screen", acquired);
        d.failed = true;
        return false;
    }
    if (acquired == VK_SUBOPTIMAL_KHR) d.stale = true;
    f.done.Reset();
    const Vulkan::vk::CommandBuffer cmd(f.buffers[0], dev.GetDispatchLoader());
    cmd.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    const VkClearValue clear{.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
    cmd.BeginRenderPass(
        {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = *d.ui_pass,
            .framebuffer = *d.ui_framebuffers[index],
            .renderArea = {{0, 0}, d.extent},
            .clearValueCount = 1,
            .pClearValues = &clear,
        },
        VK_SUBPASS_CONTENTS_INLINE);
    {
        std::lock_guard queue_lock(QueueLock()); // (ImGui uploads its textures with a submission and a queue wait)
        draw(*cmd);
    }
    cmd.EndRenderPass();
    cmd.End();
    const VkSemaphore wait = *f.acquired, signal = *d.rendered[index];
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSubmitInfo submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait,
        .pWaitDstStageMask = &stage,
        .commandBufferCount = 1,
        .pCommandBuffers = cmd.address(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal,
    };
    const VkSwapchainKHR swapchain = *d.swapchain;
    const VkPresentInfoKHR present{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &signal,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &index,
        .pResults = nullptr,
    };
    VkResult presented;
    {
        std::lock_guard queue_lock(QueueLock());
        if (dev.GetGraphicsQueue().Submit(submit, *f.done) != VK_SUCCESS) Fatal("nvn: window: vkQueueSubmit failed");
        presented = dev.GetPresentQueue().Present(present);
    }
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) d.stale = true;
    return true;
}

void EndUi() {
    if (!Vk().surface) return;
    Display& d = D();
    std::lock_guard lock(d.mu);
    {
        std::lock_guard queue_lock(QueueLock());
        Vk().device->GetLogical().WaitIdle();
    }
    d.ui_framebuffers.clear();
    d.ui_views.clear();
    d.ui_pass = {};
    d.stale = true; // the game's first frame makes a new swapchain: with the present mode (VSync) chosen on the screen
}

} // namespace rt::nvn
