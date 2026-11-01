// platform.h on Linux (POSIX plus a few Linux calls: memfd, /proc, tgkill).
#include "platform.h"

#include <dirent.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cstring>
#include <exception>
#include <filesystem>
#include <thread>

namespace rt::platform {

void InitProcess() {}

// --- Guest memory ------------------------------------------------------------

namespace {
int g_device_fd = -1;
uintptr_t g_window_lo, g_window_hi; // the guest window (the crash report names guest addresses)
std::atomic<bool (*)(uintptr_t)> g_write_hook{nullptr}; // writetrack: a write to a page it protected

/// A protection change the window cannot do without: the window's state would be wrong from here on.
void SetWindowAccess(uint8_t* p, size_t size, int prot) {
    if (mprotect(p, size, prot) == 0) return;
    const int e = errno;
    std::fprintf(stderr, "[rt] FATAL: guest window: mprotect(%p, 0x%zx) failed: %s%s\n", static_cast<void*>(p), size,
                 strerror(e), e == ENOMEM ? " (too many mappings? see /proc/sys/vm/max_map_count)" : "");
    std::fflush(stderr);
    _exit(70);
}
} // namespace

uint8_t* ReserveWindow(size_t size, uintptr_t at) {
    // Inaccessible and uncommitted: CommitWindow opens what the guest maps; Linux allocates a page when it is
    // first touched.
    void* p = mmap(reinterpret_cast<void*>(at), size, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    if (reinterpret_cast<uintptr_t>(p) != at) { // a kernel older than 4.17 takes the address as a hint only
        munmap(p, size);
        return nullptr;
    }
    g_window_lo = at;
    g_window_hi = at + size;
    return static_cast<uint8_t*>(p);
}

void CommitWindow(uint8_t* p, size_t size) {
    if (size) SetWindowAccess(p, size, PROT_READ | PROT_WRITE);
}

bool ProtectReadOnly(uint8_t* p, size_t size) { return mprotect(p, size, PROT_READ) == 0; }

bool ProtectReadWrite(uint8_t* p, size_t size) { return mprotect(p, size, PROT_READ | PROT_WRITE) == 0; }

void SetWriteFaultHook(bool (*hook)(uintptr_t a)) { g_write_hook.store(hook, std::memory_order_release); }

void ZeroWindow(uint8_t* p, size_t size) {
    if (!size) return;
    madvise(p, size, MADV_DONTNEED);
    SetWindowAccess(p, size, PROT_NONE);
}

uint8_t* CreateDeviceBacking(size_t size) {
    g_device_fd = memfd_create("guest-device-memory", MFD_CLOEXEC);
    if (g_device_fd < 0 || ftruncate(g_device_fd, static_cast<off_t>(size)) != 0) return nullptr;
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, g_device_fd, 0);
    return p == MAP_FAILED ? nullptr : static_cast<uint8_t*>(p);
}

bool MapDeviceView(uint8_t* p, size_t size, uint64_t offset) {
    return mmap(p, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, g_device_fd, static_cast<off_t>(offset)) !=
           MAP_FAILED;
}

void UnmapDeviceView(uint8_t* p, size_t size) {
    mmap(p, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
}

void* AllocateSparseZeroed(size_t size) {
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}

// --- Host files ----------------------------------------------------------------

File& File::operator=(File&& other) noexcept {
    if (this != &other) {
        Close();
        handle_ = other.handle_;
        other.handle_ = -1;
    }
    return *this;
}

bool File::Open(const std::filesystem::path& path, Mode mode) {
    Close();
    const int flags = mode == Mode::kRead ? O_RDONLY : mode == Mode::kReadWrite ? O_RDWR : O_CREAT | O_WRONLY;
    handle_ = open(path.c_str(), flags | O_CLOEXEC, 0644);
    return handle_ >= 0;
}

void File::Close() {
    if (handle_ >= 0) close(static_cast<int>(handle_));
    handle_ = -1;
}

int64_t File::ReadAt(void* out, size_t size, uint64_t offset) const {
    size_t done = 0;
    while (done < size) {
        const ssize_t r = pread(static_cast<int>(handle_), static_cast<char*>(out) + done, size - done,
                                static_cast<off_t>(offset + done));
        if (r < 0) return -1;
        if (r == 0) break; // end of file
        done += static_cast<size_t>(r);
    }
    return static_cast<int64_t>(done);
}

int64_t File::WriteAt(const void* data, size_t size, uint64_t offset) {
    size_t done = 0;
    while (done < size) {
        const ssize_t r = pwrite(static_cast<int>(handle_), static_cast<const char*>(data) + done, size - done,
                                 static_cast<off_t>(offset + done));
        if (r <= 0) return -1;
        done += static_cast<size_t>(r);
    }
    return static_cast<int64_t>(done);
}

bool File::Flush() {
    return fsync(static_cast<int>(handle_)) == 0;
}

bool File::SetSize(uint64_t size) {
    return ftruncate(static_cast<int>(handle_), static_cast<off_t>(size)) == 0;
}

int64_t File::Size() const {
    return lseek(static_cast<int>(handle_), 0, SEEK_END);
}

// --- Threads and diagnostics -------------------------------------------------

void SleepUntil(std::chrono::steady_clock::time_point deadline) {
    std::this_thread::sleep_until(deadline);
}

void AttachOutput(const std::string&, bool) {}

std::string ExecutablePath() {
    std::error_code ec;
    return std::filesystem::read_symlink("/proc/self/exe", ec).string();
}

struct Child {
    pid_t pid;
};

Child* StartChild(const std::vector<std::string>& args, const std::string& log, std::string* error,
                  const std::vector<std::string>& env) {
    std::vector<std::string> vars; // this process's, with `env`'s replacing those of the same name
    for (char** e = environ; *e; ++e) {
        const std::string v = *e, name = v.substr(0, v.find('='));
        bool replaced = false;
        for (const std::string& x : env) replaced |= x.substr(0, x.find('=')) == name;
        if (!replaced) vars.push_back(v);
    }
    vars.insert(vars.end(), env.begin(), env.end());
    std::vector<char*> envp;
    for (std::string& v : vars) envp.push_back(v.data());
    envp.push_back(nullptr);
    std::vector<char*> argv;
    for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    pid_t pid = 0;
    const int r = posix_spawn(&pid, argv[0], &fa, nullptr, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&fa);
    if (r != 0) {
        *error = std::strerror(r);
        return nullptr;
    }
    return new Child{pid};
}

bool ChildDone(Child* c, int* code) {
    int status = 0;
    if (waitpid(c->pid, &status, WNOHANG) != c->pid) return false;
    *code = WIFSIGNALED(status) ? -WTERMSIG(status) : WEXITSTATUS(status);
    return true;
}

void KillChild(Child* c) {
    kill(c->pid, SIGKILL);
    int status = 0;
    waitpid(c->pid, &status, 0);
}

void FreeChild(Child* c) {
    delete c;
}

uint64_t FreeMemory() {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    }
    std::fclose(f);
    return kb * 1024;
}

uint64_t CurrentThreadId() {
    return static_cast<uint64_t>(syscall(SYS_gettid));
}

void SetCurrentThreadName(const char* name) {
    char b[16]; // Linux: 15 characters and the terminator
    std::snprintf(b, sizeof b, "%s", name);
    pthread_setname_np(pthread_self(), b);
}

namespace {

void (*g_dump_request)() = nullptr;

void WriteStderr(const void* p, size_t n) {
    if (write(2, p, n) < 0) {} // diagnostics only
}

void OnDumpSignal(int) {
    if (g_dump_request) g_dump_request();
}

/// SIGUSR2, sent to one thread at a time by DumpHostStacks: that thread's stack.
void DumpThisThread(int) {
    char b[64];
    const int n = std::snprintf(b, sizeof b, "[rt] --- host tid %ld\n", (long)syscall(SYS_gettid));
    WriteStderr(b, static_cast<size_t>(n));
    void* frames[48];
    const int count = backtrace(frames, 48);
    backtrace_symbols_fd(frames, count, 2);
}

// --- Crash report -------------------------------------------------------------------------------------------

void (*g_describe)() = nullptr;
std::atomic<bool> g_crashing{false};

/// Everything a crash report says after its first line (this thread's stack, the runtime's guest state); then
/// the process ends (exit code 71). Not async-signal-safe beyond backtrace: acceptable once the process is lost.
[[noreturn]] void FinishCrash() {
    char b[64];
    const int n = std::snprintf(b, sizeof b, "[rt]   host tid %ld, stack:\n", (long)syscall(SYS_gettid));
    WriteStderr(b, static_cast<size_t>(n));
    void* frames[48];
    backtrace_symbols_fd(frames, backtrace(frames, 48), 2);
    if (g_describe) g_describe();
    std::fflush(stderr);
    _exit(71);
}

void OnCrashSignal(int sig, siginfo_t* info, void*) {
    if (sig == SIGSEGV && info) { // write tracking first: returning retries the instruction
        const uintptr_t at = reinterpret_cast<uintptr_t>(info->si_addr);
        auto* hook = g_write_hook.load(std::memory_order_acquire);
        if (hook && at >= g_window_lo && at < g_window_hi && hook(at)) return;
    }
    if (g_crashing.exchange(true)) _exit(71);
    char b[160];
    const bool fault = sig == SIGSEGV || sig == SIGBUS || sig == SIGILL || sig == SIGFPE; // si_addr is meaningful
    const int n = fault ? std::snprintf(b, sizeof b, "[rt] CRASH: signal %d (%s), address %p\n", sig, strsignal(sig),
                                        info ? info->si_addr : nullptr)
                        : std::snprintf(b, sizeof b, "[rt] CRASH: signal %d (%s)\n", sig, strsignal(sig));
    WriteStderr(b, static_cast<size_t>(n));
    const uintptr_t a = info ? reinterpret_cast<uintptr_t>(info->si_addr) : 0;
    if ((sig == SIGSEGV || sig == SIGBUS) && a >= g_window_lo && a < g_window_hi) {
        const int m = std::snprintf(b, sizeof b, "[rt]   guest address 0x%llx: not mapped by the guest\n",
                                    static_cast<unsigned long long>(a - g_window_lo));
        WriteStderr(b, static_cast<size_t>(m));
    }
    FinishCrash();
}

void OnTerminate() {
    if (g_crashing.exchange(true)) _exit(71);
    std::fprintf(stderr, "[rt] CRASH: std::terminate (an exception nobody caught, or one thrown through a function "
                         "that may not throw)\n");
    FinishCrash();
}

} // namespace

void InstallCrashReport(void (*describe)()) {
    g_describe = describe;
    std::set_terminate(OnTerminate);
    struct sigaction sa{};
    sa.sa_sigaction = OnCrashSignal;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    for (int sig : {SIGBUS, SIGILL, SIGFPE, SIGABRT}) sigaction(sig, &sa, nullptr);
    // SIGSEGV stays installed: write tracking's faults come back here again and again (a fault in the crash
    // report itself still ends in g_crashing's _exit).
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
}

void InstallDumpTrigger(void (*request)()) {
    g_dump_request = request;
    signal(SIGUSR1, OnDumpSignal);
    signal(SIGUSR2, DumpThisThread);
}

void DumpHostStacks() {
    const pid_t self = static_cast<pid_t>(syscall(SYS_gettid));
    if (DIR* d = opendir("/proc/self/task")) {
        while (dirent* e = readdir(d)) {
            const long tid = std::atol(e->d_name);
            if (tid > 0 && tid != self) {
                syscall(SYS_tgkill, getpid(), tid, SIGUSR2);
                usleep(20000); // keep the dumps from interleaving
            }
        }
        closedir(d);
    }
    DumpThisThread(0);
}

void LockStderr() {
    flockfile(stderr);
}

void UnlockStderr() {
    funlockfile(stderr);
}

} // namespace rt::platform
