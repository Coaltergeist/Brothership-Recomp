/* freestanding stand-in (README.txt) */
#pragma once
typedef long clock_t;
#define CLOCKS_PER_SEC 1000L
struct timespec { long long tv_sec; long tv_nsec; };
clock_t clock(void);
int nanosleep(const struct timespec* t, struct timespec* rest);
