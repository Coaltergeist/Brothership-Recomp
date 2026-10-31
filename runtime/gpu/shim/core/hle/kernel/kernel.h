// Shim: audio_core passes System::Kernel() to KEvent/KProcess calls and never looks inside.
#pragma once

namespace Kernel {
class KernelCore {};
} // namespace Kernel
