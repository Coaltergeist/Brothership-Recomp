// The NVN tap: RT_NVN_TAP=1 logs every NVN call while the
// game's own NVN driver (lifted) keeps rendering.
//
// The game gets every NVN function from nvnBootstrapLoader or
// nvnDeviceGetProcAddress by name. The tap overrides nvnBootstrapLoader (hle.h)
// and hands out, for each function pointer either returns, a host wrapper
// that records the call and runs the real function. The wrapper of
// nvnDeviceGetProcAddress wraps what it returns in turn.
//
// Output, rewritten every 10 s in the working directory (the benchmark ends a
// run with a signal):
// - nvn_census.tsv: per function, in the order of first call: calls, frames
//   with calls, calls per such frame (mean, max), first and last call (frame,
//   seconds), the threads that called it.
// - nvn_args.tsv: for the functions in the tap's argument table (texture,
//   sampler, memory and queue setup, draws), how often each combination of the
//   recorded arguments was passed.
#pragma once

#include "../rt.h"

#include <string>

namespace rt::nvn::tap {

/// Installs the tap if RT_NVN_TAP is set. After hle::Init.
void Init();

/// Shadow oracle: after each tapped call of `name`, `fn` gets
/// the registers before and after the real function, to run an own
/// implementation on `before` and report any difference. For functions that
/// only compute from their arguments (sizes, alignments, handles, getters).
/// Several shadows of one function run in the order they were added.
using ShadowFn = void (*)(const std::string& name, const GuestContext& before, const GuestContext& after);
void AddShadow(const std::string& name, ShadowFn fn);

} // namespace rt::nvn::tap
