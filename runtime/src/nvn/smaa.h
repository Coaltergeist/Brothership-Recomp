// SMAA 1x on the presented image (RT_AA=smaa, a player option: options.h), at the image's own size before it is
// scaled to the window, as LASRecomp and yuzu do; the overlay is drawn afterwards, unfiltered. Headless runs show
// nothing, so they have none (captures read the game's image).
#pragma once

#include "images.h"

namespace rt::nvn {

/// RT_AA: off (default) or smaa; stops the run for another value.
bool SmaaOn();

/// Records SMAA of `input` (level 0, `extent` texels, in VK_IMAGE_LAYOUT_GENERAL after everything that wrote it)
/// into `cmd` for frame slot `slot` (the slot's previous frame has finished: its descriptors and input view are
/// reused). Returns the image to show instead: the same size and format, in VK_IMAGE_LAYOUT_GENERAL, the writes
/// done (barrier) before the caller reads it.
VkImage SmaaApply(Vulkan::vk::CommandBuffer cmd, u32 slot, const TextureImage& input, VkExtent2D extent);

} // namespace rt::nvn
