// The game module's C library (BUILD_PLAN Phase 2): the few functions the lifted code and drippu's recomp_runtime.c
// name, for a module built without one (tools/build_game.sh; freestanding/README.txt). Module builds only: the
// static build links the real C library.
//
// Memory and string functions are real: the compiler emits calls to memcpy / memset for large copies. printf and
// fprintf go to the runtime's log (NxHost.print: drippu's "unhandled instruction" message). Files, clocks and sleeps
// are only on drippu's standalone paths - its save files, its own run loop and SVC handler, clock fallbacks - which our
// runtime never takes (it supplies the SVCs and every RecompHostMem callback): they fail, or stop the run.
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include "game.h"

extern const NxHost* g_nx_host; // entry.c

#undef memcpy
#undef memmove
#undef memset
#undef memcmp

void* memcpy(void* d, const void* s, size_t n) {
    unsigned char* o = d;
    const unsigned char* i = s;
    while (n--) *o++ = *i++;
    return d;
}

void* memmove(void* d, const void* s, size_t n) {
    unsigned char* o = d;
    const unsigned char* i = s;
    if (o < i) {
        while (n--) *o++ = *i++;
    } else {
        while (n--) o[n] = i[n];
    }
    return d;
}

void* memset(void* d, int c, size_t n) {
    unsigned char* o = d;
    while (n--) *o++ = (unsigned char)c;
    return d;
}

int memcmp(const void* a, const void* b, size_t n) {
    const unsigned char *x = a, *y = b;
    for (; n; --n, ++x, ++y) {
        if (*x != *y) return *x < *y ? -1 : 1;
    }
    return 0;
}

int bcmp(const void* a, const void* b, size_t n) { return memcmp(a, b, n); }

size_t strlen(const char* s) {
    size_t n = 0;
    while (s[n]) ++n;
    return n;
}

char* strrchr(const char* s, int c) {
    const char* last = 0;
    for (;; ++s) {
        if (*s == (char)c) last = s;
        if (!*s) return (char*)last;
    }
}

// printf / fprintf: the runtime's log (both streams)
typedef struct FILE FILE;
int printf(const char* fmt, ...) {
    va_list a;
    va_start(a, fmt);
    g_nx_host->print(fmt, a);
    va_end(a);
    return 0;
}
int fprintf(FILE* f, const char* fmt, ...) {
    (void)f;
    va_list a;
    va_start(a, fmt);
    g_nx_host->print(fmt, a);
    va_end(a);
    return 0;
}
int puts(const char* s) {
    printf("%s\n", s);
    return 0;
}

// drippu's standalone paths only (above): failures
int snprintf(char* s, size_t n, const char* fmt, ...) { (void)fmt; if (n) s[0] = 0; return 0; }
FILE* fopen(const char* path, const char* mode) { (void)path, (void)mode; return 0; }
int fclose(FILE* f) { (void)f; return 0; }
size_t fread(void* p, size_t size, size_t n, FILE* f) { (void)p, (void)size, (void)n, (void)f; return 0; }
size_t fwrite(const void* p, size_t size, size_t n, FILE* f) { (void)p, (void)size, (void)n, (void)f; return 0; }
int fseek(FILE* f, long off, int whence) { (void)f, (void)off, (void)whence; return -1; }
long ftell(FILE* f) { (void)f; return -1; }
int fflush(FILE* f) { (void)f; return 0; }
int remove(const char* path) { (void)path; return -1; }
long clock(void) { return 0; }
void* calloc(size_t n, size_t size) { (void)n, (void)size; return 0; }
void* malloc(size_t size) { (void)size; return 0; }
void free(void* p) { (void)p; }
_Noreturn void abort(void) { __builtin_trap(); }
struct timespec;
int nanosleep(const struct timespec* t, struct timespec* rest) { (void)t, (void)rest; return 0; }
#ifdef _WIN32
void Sleep(unsigned long ms) { (void)ms; }
int _mkdir(const char* path) { (void)path; return -1; }

// What Microsoft's C runtime gives a DLL: the marker for floating-point use, the stack probe the compiler calls in
// frames over 4 KiB (it touches each page so the guard page grows the stack in order; LLVM compiler-rt's
// chkstk.S), and the TLS directory for rt_trace_pc (mingw-w64's tlssup.c, public domain).
int _fltused = 0;

// The DLL's entry point, and one import. Windows' loader gives a DLL that imports nothing no TLS slot of its own: its
// _tls_index stays 0, the executable's, and rt_trace_pc landed in the runtime's thread-local block (2026-10-31: the
// runtime's jump cache overwritten, a crash at the first indirect jump; a small test DLL showed it - index 0 without
// imports, its own with one KERNEL32 import, with our TLS directory or Microsoft's tlssup.obj alike). So the module
// imports GetCurrentThreadId (never called), from an import library made from kernel32.def (tools/build_game.sh:
// llvm-dlltool), not Microsoft's.
__declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);

int __stdcall _DllMainCRTStartup(void* module, unsigned long reason, void* reserved) {
    (void)module, (void)reserved;
    if (reason == 0xFFFFFFFFu) GetCurrentThreadId(); // (no such reason: the import, kept)
    return 1;
}

__attribute__((naked)) void __chkstk(void) {
    __asm__ volatile("push %rcx\n"
                     "push %rax\n"
                     "cmp $0x1000, %rax\n"
                     "lea 24(%rsp), %rcx\n"
                     "jb 1f\n"
                     "2:\n"
                     "sub $0x1000, %rcx\n"
                     "test %rcx, (%rcx)\n"
                     "sub $0x1000, %rax\n"
                     "cmp $0x1000, %rax\n"
                     "ja 2b\n"
                     "1:\n"
                     "sub %rax, %rcx\n"
                     "test %rcx, (%rcx)\n"
                     "pop %rax\n"
                     "pop %rcx\n"
                     "ret\n");
}

typedef void (*TlsCallback)(void* module, unsigned long reason, void* reserved);
typedef struct {
    uint64_t start, end, index, callbacks;
    uint32_t zero_fill, characteristics;
} TlsDirectory;
unsigned long _tls_index = 0;
__attribute__((section(".tls"))) char _tls_start = 0;
__attribute__((section(".tls$ZZZ"))) char _tls_end = 0;
__attribute__((section(".CRT$XLA"))) TlsCallback __xl_a = 0;
__attribute__((section(".CRT$XLZ"))) TlsCallback __xl_z = 0;
const TlsDirectory _tls_used = {(uint64_t)&_tls_start, (uint64_t)&_tls_end, (uint64_t)&_tls_index,
                                (uint64_t)(&__xl_a + 1), 0, 0};
#else
int mkdir(const char* path, unsigned mode) { (void)path, (void)mode; return -1; }
#endif
