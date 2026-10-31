// Shim: audio_core's pool mapper only names the current-process pseudo-handle.
#pragma once
#include "common/common_types.h"

namespace Kernel::Svc {
constexpr u32 CurrentProcess = 0xFFFF8001;
} // namespace Kernel::Svc
