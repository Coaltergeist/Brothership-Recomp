// The NVN layer's image on the host window.
#pragma once

#include <functional>

#include "texture.h"
#include "vk.h"

namespace rt::nvn {

/// Shows texture `t` (the texture the game presented) in the host window, scaled to fit with black bars,
/// with the overlay's text box on top. Nothing without a window (RT_HEADLESS=1).
void ShowInWindow(const TextureState& t);

/// The start screen's drawing, before the game's first frame: a swapchain in plain (UNORM) colour
/// that can be drawn into, a render pass that clears it to black, a framebuffer per swapchain image. UiTarget gives
/// the render pass, the image count and format (ImGui's pipeline is made for them); ShowUi records `draw` inside
/// that render pass (with QueueLock held: ImGui uploads textures with a submission of its own) and presents, giving
/// the swapchain's size - false when nothing was shown (no window area, out of date: call again). EndUi waits for
/// the GPU and drops the render pass; the game's first present makes its own swapchain (sRGB, as the game's).
bool UiTarget(VkRenderPass* pass, u32* images, VkFormat* format);
bool ShowUi(const std::function<void(VkCommandBuffer)>& draw, VkExtent2D* extent);
void EndUi();

} // namespace rt::nvn
