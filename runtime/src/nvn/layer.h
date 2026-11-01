// The NVN layer: Nintendo's graphics API implemented by the runtime
// on Vulkan, in place of the game's own NVN driver.
//
// RT_GPU=nvn turns it on (the default, RT_GPU=driver, keeps the game's driver
// on top of video_core). The layer overrides nvnBootstrapLoader (hle.h) and
// answers nvnDeviceGetProcAddress from its table of every NVN name of SDK
// 3.5.0 to 12.3.2 (names.inc); a function it does not implement yet stops the
// run with its name. Each function's first call is logged ("nvn: first call"),
// in the order the tap's census lists them (tap.h).
//
// RT_NVN_DIFFTEST=1 runs the layer's pure functions (state objects, builders)
// next to the game's own SDK functions at the first NVN call, logs every
// difference and ends the run (difftest.cpp).
#pragma once

#include "../rt.h"

namespace rt::nvn {

/// RT_GPU=nvn, or RT_GPU=auto (the default) and the program loads NVN. After the symbols are loaded.
bool Enabled();

/// Installs the layer if enabled. After hle::Init, before guest code runs.
void Init();

/// Shadow checks for the tap (tap.h): compares the layer's answers with the
/// real driver's while the real driver runs. Called by the tap's Init.
void InstallShadows();

/// Writes the driver's pipeline cache if pipelines were made since its last write, and logs the run's
/// pipeline totals (pipeline.cpp). Called by Exit; nothing without a cache.
void SavePipelineCacheAtExit();

/// From now on this run writes the pipeline cache file neither at exit nor at a quiet moment: a run that follows
/// replaces it (the start screen handing the seed over to `nxrt --seed`: its own copy is the old one).
void KeepPipelineCacheFile();

} // namespace rt::nvn
