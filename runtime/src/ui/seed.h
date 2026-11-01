// The shader cache's own run (`nxrt --seed`): Build's shader steps (build.h Steps::kShaders)
// in a process with no Vulkan device of its own - a seed made beside a running start screen left most of the stages
// out of the pipeline cache. A window in the start screen's look (look.h), drawn by SDL's
// software renderer: the progress, the time left, Cancel (or closing the window); a failure stays on screen until
// Back. Then the start screen again, on its Build page (`--after-seed`). Without a window (no display) the steps run
// all the same.
#pragma once

#include <string>

namespace rt::ui {

/// `nxrt --seed`: the process's exit code (0 when the start screen was started again).
int RunSeed();

/// Starts this program again with `arg` (one argument, or none when empty), outliving this process; false (why) when
/// it could not be started.
bool StartSelf(const char* arg, std::string* why);

} // namespace rt::ui
