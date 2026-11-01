// Stand-in for <Windows.h> as ooz's stdafx.h uses it, on other systems than Windows (seed/ooz.cpp). Only its
// command-line program (never called here) loads Oodle's DLL and reads the clock.
#pragma once

#include <stdint.h>
#include <string.h>
#include <sys/stat.h> // (its command-line program: stat)

#define __int64 long long
#define __forceinline inline __attribute__((always_inline))
#define WINAPI

typedef void* HINSTANCE;
typedef union {
    struct {
        uint32_t LowPart;
        int32_t HighPart;
    };
    long long QuadPart;
} LARGE_INTEGER;

static inline HINSTANCE LoadLibraryA(const char*) { return 0; }
static inline void* GetProcAddress(HINSTANCE, const char*) { return 0; }
static inline int QueryPerformanceCounter(LARGE_INTEGER* t) { t->QuadPart = 0; return 1; }
static inline int QueryPerformanceFrequency(LARGE_INTEGER* t) { t->QuadPart = 1; return 1; }
