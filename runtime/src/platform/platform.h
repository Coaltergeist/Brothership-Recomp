// The host operating system as the runtime uses it: guest memory, host files,
// thread ids and names, host stack dumps, whole-line logging. Everything else
// in runtime/src is portable C++; one implementation of this file per OS
// (posix.cpp, windows.cpp).
//
// Not covered here yet, for the Windows port:
// - GNU extensions (__atomic builtins, the cmpxchg16b asm in rt.cpp, format
//   attributes) assume GCC or clang: build the runtime with clang-cl there.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>

namespace rt::platform {

/// Process-wide settings, first thing in main. Windows: 1 ms timer resolution (sleeps and timed waits round
/// to 15.6 ms otherwise), no power throttling (EcoQoS, timer resolution ignored while the window is hidden),
/// UTF-8 console output. POSIX: nothing.
void InitProcess();

// --- Guest memory ------------------------------------------------------------
//
// The guest address space is one host reservation (the "window") at a fixed
// host address. Only ranges the guest has mapped are accessible (CommitWindow);
// everything else faults, so a stray access is reported instead of reading zero.
// Pages are private and read as zero until written,
// except those aliased onto the device backing (MapDeviceView), which the GPU
// reads as "physical" memory.

/// Reserves `size` bytes of address space for the guest window at host address `at`, all inaccessible;
/// null if that range is not free.
uint8_t* ReserveWindow(size_t size, uintptr_t at);
/// Makes [p, p + size) of the window usable (read-write; zero-filled where it was
/// never written or was zeroed). POSIX opens the pages; Windows commits here.
void CommitWindow(uint8_t* p, size_t size);
/// [p, p + size) is unmapped: it reads as zero once committed again and is
/// inaccessible until then. POSIX gives the host pages back; Windows zero-fills
/// small ranges in place (cheaper than remapping) and gives large ones back.
void ZeroWindow(uint8_t* p, size_t size);

/// Creates the device backing (`size` bytes, shared) and returns its own view.
uint8_t* CreateDeviceBacking(size_t size);
/// Replaces [p, p + size) of the window by the device backing's bytes
/// [offset, offset + size): both views then show the same memory.
bool MapDeviceView(uint8_t* p, size_t size, uint64_t offset);
/// Turns [p, p + size) of the window back into private zero pages.
void UnmapDeviceView(uint8_t* p, size_t size);

/// Write tracking (writetrack.cpp): makes [p, p + size) of the window read-only /
/// read-write again. False when the host refuses (Windows: a range across separately mapped views, or memory
/// that is not a mapped view: the caller tries page by page).
bool ProtectReadOnly(uint8_t* p, size_t size);
bool ProtectReadWrite(uint8_t* p, size_t size);
/// The function the fault handlers ask first about a fault at host address `a` in the window (writetrack's
/// OnWriteFault): true = handled, retry the instruction. Windows asks for writes only; POSIX for every SIGSEGV.
void SetWriteFaultHook(bool (*hook)(uintptr_t a));

/// Zero-filled memory whose host pages are only allocated when touched (the
/// fastmem page table: 1 GB, written sparsely).
void* AllocateSparseZeroed(size_t size);

// --- Host files ----------------------------------------------------------------

/// A host file read and written at explicit offsets (safe from several threads).
class File {
public:
    enum class Mode { kRead, kReadWrite, kCreate }; // kCreate: create (or open) for writing
    File() = default;
    File(File&& other) noexcept : handle_{other.handle_} { other.handle_ = -1; }
    File& operator=(File&& other) noexcept;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() { Close(); }

    bool Open(const std::filesystem::path& path, Mode mode);
    bool IsOpen() const { return handle_ != -1; }
    void Close();
    /// Bytes read (fewer at the end of the file), or -1.
    int64_t ReadAt(void* out, size_t size, uint64_t offset) const;
    /// Bytes written, or -1.
    int64_t WriteAt(const void* data, size_t size, uint64_t offset);
    bool Flush();
    bool SetSize(uint64_t size);
    int64_t Size() const;

private:
    intptr_t handle_ = -1; // a file descriptor, or a HANDLE on Windows
};

// --- Threads and diagnostics -------------------------------------------------

/// Sleeps until `deadline` (steady clock), to well under a millisecond: the guest paces frames and audio
/// with short sleeps, and std::this_thread's sleeps on Windows round up to whole milliseconds and wake on
/// the next timer tick (~1.5 ms late on average, which made a 30 fps game run at 29). Windows: a
/// high-resolution waitable timer per thread (std::this_thread if it cannot be made). POSIX: std::this_thread.
void SleepUntil(std::chrono::steady_clock::time_point deadline);
inline void SleepFor(std::chrono::nanoseconds duration) { SleepUntil(std::chrono::steady_clock::now() + duration); }

/// Where stdout and stderr go when the program has none (Windows, a build without a console window,
/// started by a double-click): the console of the terminal it was started from, if any, else `log_file` (the previous
/// one kept as <name>.old.log; `continuing`: added to instead - a run that carries on its starter's, as the start
/// screen's hand-over to `--seed` and back, ui/seed.h). Output that already goes somewhere (a console, a redirect:
/// the bench scripts) is left alone. POSIX: nothing.
void AttachOutput(const std::string& log_file, bool continuing = false);

/// A child process (the game builder's tools, build/build.h): `args` (UTF-8; args[0] the program's path), its stdout and
/// stderr to the file `log` (made or emptied), no stdin, no window of its own (Windows: CREATE_NO_WINDOW, so a program
/// without a console starts its 573 compile jobs invisibly; SDL's processes do that only in a mode that loses the
/// exit code), and nothing else of ours inherited. Null (`error`: why) when it could not start.
/// `env`: NAME=value entries added to (or replacing in) this process's environment for the child only.
struct Child;
Child* StartChild(const std::vector<std::string>& args, const std::string& log, std::string* error,
                  const std::vector<std::string>& env = {});
/// Whether it has ended (without waiting); `code`: its exit code - Windows: the process's (0xC... for a crash),
/// POSIX: the status, or -signal when a signal ended it.
bool ChildDone(Child* c, int* code);
/// Ends it (if running) and waits for it.
void KillChild(Child* c);
/// Frees it (after ChildDone said so, or KillChild).
void FreeChild(Child* c);

/// This program's own file (UTF-8): what the builder's seed stage and a restart into the game start again, whatever
/// the release names it (BrothershipRecomp.exe; developer builds: nxrt).
std::string ExecutablePath();

/// The physical memory free for new processes now, in bytes (Windows: available physical memory; Linux:
/// MemAvailable); 0 when unknown. The game builder's job count (build.h).
uint64_t FreeMemory();

/// The OS id of the calling thread (as debuggers and profilers show it).
uint64_t CurrentThreadId();
/// Names the calling thread for debuggers and profilers (truncated as the OS needs).
void SetCurrentThreadName(const char* name);

/// Reports a crash in the log before the process dies (exit code 71): an unhandled fault (access violation,
/// illegal instruction ... — after the platform's own guest-memory handling), std::terminate (an exception
/// nobody caught) or abort(). The report: what and where, the crashing thread's host stack (Windows:
/// "module+0xoffset" from the unwind tables; Linux: backtrace), then `describe()` (the runtime's guest state).
/// Windows also writes nxrt_crash.dmp (a minidump for WinDbg / Visual Studio) and shows no error dialog.
void InstallCrashReport(void (*describe)());

/// Calls `request` (from a signal or console handler: it must only set a flag)
/// when the user asks for a thread dump. POSIX: kill -USR1 <pid>.
void InstallDumpTrigger(void (*request)());
/// Writes every host thread's id and native stack to stderr (resolve frames with
/// addr2line on Linux). Called from an ordinary thread, not from a handler.
void DumpHostStacks();

/// Whole-line logging: guest threads log in parallel.
void LockStderr();
void UnlockStderr();

} // namespace rt::platform
