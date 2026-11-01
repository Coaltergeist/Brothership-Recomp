/* freestanding stand-in (README.txt) */
#pragma once
#include <stddef.h>
void* memcpy(void* d, const void* s, size_t n);
void* memmove(void* d, const void* s, size_t n);
void* memset(void* d, int c, size_t n);
int memcmp(const void* a, const void* b, size_t n);
size_t strlen(const char* s);
char* strrchr(const char* s, int c);
#define memcpy(d, s, n) __builtin_memcpy(d, s, n)
#define memmove(d, s, n) __builtin_memmove(d, s, n)
#define memset(d, c, n) __builtin_memset(d, c, n)
#define memcmp(a, b, n) __builtin_memcmp(a, b, n)
