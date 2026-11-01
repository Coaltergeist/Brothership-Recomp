/* freestanding stand-in (README.txt): drippu's standalone save files and messages */
#pragma once
#include <stdarg.h>
#include <stddef.h>
typedef struct FILE FILE;
#define stdout ((FILE*)1)
#define stderr ((FILE*)2)
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
FILE* fopen(const char* path, const char* mode);
int fclose(FILE* f);
size_t fread(void* p, size_t size, size_t n, FILE* f);
size_t fwrite(const void* p, size_t size, size_t n, FILE* f);
int fseek(FILE* f, long off, int whence);
long ftell(FILE* f);
int fflush(FILE* f);
int printf(const char* fmt, ...);
int fprintf(FILE* f, const char* fmt, ...);
int snprintf(char* s, size_t n, const char* fmt, ...);
int puts(const char* s);
int remove(const char* path);
