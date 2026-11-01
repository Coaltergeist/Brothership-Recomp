/* freestanding stand-in (README.txt) */
#pragma once
#include <stddef.h>
void* calloc(size_t n, size_t size);
void* malloc(size_t size);
void free(void* p);
_Noreturn void abort(void);
