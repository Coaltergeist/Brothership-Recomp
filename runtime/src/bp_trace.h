// Brothership's Blueprint calls, logged by name (bp_trace.cpp): RT_BP_TRACE.
#pragma once

namespace rt::bp_trace {

/// Installs the trace if RT_BP_TRACE is set and the game's code is 1.0.1's (else logs why not). After hle::Init,
/// before guest code runs.
void Install();

} // namespace rt::bp_trace
