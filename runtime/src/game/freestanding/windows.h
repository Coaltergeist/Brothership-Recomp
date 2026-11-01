/* freestanding stand-in (README.txt) */
#pragma once
typedef unsigned long DWORD;
void Sleep(DWORD ms);
#define MemoryBarrier() __atomic_thread_fence(__ATOMIC_SEQ_CST)
