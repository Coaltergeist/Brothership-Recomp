// Thread dumps and the freeze watchdog, the same on every OS.
//
// A dump lists each guest thread from the runtime's own records (KThread):
// host thread id, kernel and nn::os names, what it waits on, its guest call
// chain (frame-pointer walk); then every host thread's native stack
// (platform::DumpHostStacks). It runs on the diagnostics thread, never in a
// signal handler.
//
// Triggers: on demand (POSIX: kill -USR1 <pid>), and the watchdog: no presented
// frame for RT_WATCHDOG seconds (default 30, 0 = off; armed after the first
// frame) logs "watchdog: no frame for ...", dumps twice 3 s apart, and re-arms
// when frames resume. The process keeps running.
#pragma once

#include <vector>

#include "rt.h"
#include "kernel.h"

namespace rt::diag {

/// Installs the dump trigger and starts the diagnostics thread.
void Start();

/// The first name-like string in the thread's nn::os::ThreadType (pointer at
/// TLS+0x1F8): the game's name for the thread, or "".
std::string OsThreadName(const KThread& t);

/// Return addresses of the guest frame-pointer chain starting at `fp`.
std::vector<u64> GuestCallChain(u64 fp, int max_depth);

} // namespace rt::diag
