// CPU write tracking for guest memory (BUFFER_CACHE_PLAN Phase 1, Windows A: page protection).
//
// A user (the NVN layer's buffer cache, Phase 2) asks whether guest pages changed since it last copied them:
//
//   writetrack::Watch(va, size);           // protect,
//   const u32 since = writetrack::Now();   // then the generation,
//   ...copy the bytes...                   // then copy (a write in between is in the copy)
//   if (!writetrack::ChangedSince(va, size, since)) { /* the copy is still good */ }
//
// Watched pages are read-only; the first write to one faults (lifted code and host code alike: the fault
// handler is process-wide), and the handler (OnWriteFault, through platform::SetWriteFaultHook) bumps the page's
// generation and makes it writable again before the write is retried. A page that is not watched at the time of
// the question counts as changed, and so does one protected again after `since` (it may have been written while
// unprotected - a hot page, say: the first Phase 2 check found such misses, 2026-10-24), so every way of missing a
// protection errs on the safe side. Writes by the OS
// kernel (a file read into guest memory) would fail on a read-only page instead of faulting: host code that hands
// guest memory to the kernel calls HostWrite first (Memory::Access with kWrite does). Remaps and zero-fills call
// Forget. A page written in each of its last 3 watch cycles is "hot": left writable (always changed) for the
// next 60, so pages the game writes every frame cost no fault every frame.
//
// RT_WRITE_TRACK=0: nothing is ever watched (every page counts as changed). RT_WRITE_TRACK_CHECK=1 (test aid): a
// page watched again with no write seen is compared with a hash taken when it was protected; a difference is a
// missed write, logged. RT_WRITE_TRACK_STATS=1: counters every 5 s. RT_WRITE_TRACK_SELFTEST=1: SelfTest() at
// startup (rt.cpp), then exit.
#pragma once

#include "rt.h"

namespace rt::writetrack {

/// Whether tracking is on (RT_WRITE_TRACK, default on).
bool On();
/// RT_WRITE_TRACK_CHECK: also verify (see the top); the NVN layer then watches every page it copies, as Phase 2
/// will, so the check runs under the game's own writes before anything relies on the answers.
bool CheckOn();
/// The current generation: pages written later get a larger one.
u32 Now();
/// Protects the pages of [va, va + size) that can be watched (mapped read-write guest memory, not excluded, not
/// hot); the others stay unwatched and count as changed.
void Watch(u64 va, u64 size);
/// Whether any page of [va, va + size) was written after `since` (Now() after its Watch), is not watched, or was
/// protected again after `since`.
bool ChangedSince(u64 va, u64 size, u32 since);
/// Host code is about to write [va, va + size) in a way that cannot fault (the OS kernel): unprotect and bump.
void HostWrite(u64 va, u64 size);
/// [va, va + size) is remapped, zero-filled or given back: unprotect what is watched and bump every page.
void Forget(u64 va, u64 size);
/// [va, va + size) is (or no longer is) also mapped elsewhere (the device backing), where writes do not fault:
/// never watched while excluded.
void Exclude(u64 va, u64 size, bool excluded);
/// The page's state, for diagnostics: "watched|free|releasing gen <g> watched_at <w> streak <s>".
std::string Describe(u64 va);
/// Installs the fault hook (once guest memory is reserved).
void Init();
/// RT_WRITE_TRACK_SELFTEST: protect, write and remap a test region every way the runtime writes guest memory;
/// logs each check and returns whether all passed.
bool SelfTest();

} // namespace rt::writetrack
