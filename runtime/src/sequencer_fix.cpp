// Unreal Sequencer repeater events at frame rates above the sequences' time grid (2026-11-04).
//
// Brothership's battle sequences count time in 30 ticks a second (their MovieScene's tick resolution; Unreal's
// default is 24000). Each frame the flat sequence updater (UE 4.27 FSequenceUpdater_Flat::Update) turns the time swept
// since the last frame, (previous, current], into the whole ticks it holds, [floor(previous) + 1, floor(current)],
// and imports the one-shot entities there: trigger keys at those ticks and the repeater sections (event "every
// frame") that span them. At 30 fps every frame crosses one tick. At 60 about every other frame crosses none, so the
// repeaters run on about half the frames: Red Shell's final kick (BS_BA_RedCarapace_03, CheckKickSuccess /
// CheckKickFail in repeaters) then misses a press - a press counts as pushed for one frame - and MoveShell, which
// moves by speed x the frame's delta time, moves the shell at about half speed (traced with RT_HLE_TRACE !bt in a
// battle: the check's call chain ran on 1-3 frame gaps at 60, every frame at 30).
//
// The fix, for forward play: when time moved but crossed no tick, the update runs with the sweep starting at the
// current tick, so the repeaters spanning it run this frame too; the trigger keys found there already fired when
// time crossed that tick, so trigger sections' imports are skipped during that update. Updates that cross a tick, and
// sequences of a finer grid (whose frames always cross one), are not touched. Only the flat updater: the sub-sequence
// updater's ranges are per sub-sequence (and the battle sequences have none).
//
// Game 1.0.1's code (guest = main ELF + 0x8004000), checked word by word before anything is installed:
//   0x9e4f7e0  FSequenceUpdater_Flat::Update(this, linker, instance handle, player, const FMovieSceneContext&),
//              reached only through its vtable; the context starts with the TRange<FFrameTime> swept, its
//              direction at +0x20 (0 = forwards); the one-shot query and import at main 0x1e4bc74 / 0x1e4bcb8
//   0x9f7a3b4  UMovieSceneEventTriggerSection's ImportEntityImpl (its keys by entity id, no time test; the
//              repeater's, main 0x1f7490c, tests that its range holds the current frame)
//   0x9f7a728  its IMovieSceneEntityProvider thunk (this - 0xe0, then a direct branch), the vtables' entry
#include "sequencer_fix.h"

#include <atomic>
#include <cstring>

#include "hle.h"
#include "rt.h"

namespace rt::sequencer_fix {
namespace {

constexpr u64 kFlatUpdate = 0x9e4f7e0;
constexpr u64 kTriggerImport = 0x9f7a3b4;
constexpr u64 kTriggerImportThunk = 0x9f7a728;

struct Expect {
    u64 va;
    u32 words[4];
};
constexpr Expect kExpected[] = {
    {kFlatUpdate, {0xd10343ff, 0xa907fbfd, 0x9101e3fd, 0xfd003be8}},
    {kTriggerImport, {0xd10583ff, 0x6d0e2beb, 0x6d0f23e9, 0xa9107bfd}},
    {kTriggerImportThunk, {0xd1038000, 0x17ffff22, 0xd10203ff, 0xa90657f6}},
};

// TRange<FFrameTime>'s bounds (ERangeBoundTypes: 0 exclusive, 1 inclusive, 2 open), as the context holds them.
struct Bound {
    u8 type, pad[3];
    s32 frame;
    float sub;
};
static_assert(sizeof(Bound) == 12);
constexpr u8 kExclusive = 0, kInclusive = 1, kOpen = 2;

thread_local bool t_skip_triggers = false; // during an update whose sweep was moved back to its tick
std::atomic<u64> g_moved{0};

void FlatUpdate(GuestContext* c, void*) {
    const u64 ctx = c->x[4];
    Bound lo, hi;
    u8 direction;
    g_mem.Read(ctx, &lo, sizeof lo);
    g_mem.Read(ctx + 12, &hi, sizeof hi);
    g_mem.Read(ctx + 0x20, &direction, 1);
    bool moved_back = false;
    if (direction == 0 && lo.type != kOpen && hi.type != kOpen) {
        // the updater's frame numbers: [first, end)
        const s32 end = hi.frame + ((hi.type == kInclusive || hi.sub != 0.0f) ? 1 : 0);
        const s32 first = lo.frame + ((lo.type == kExclusive || lo.sub != 0.0f) ? 1 : 0);
        const bool advanced = hi.frame > lo.frame || (hi.frame == lo.frame && hi.sub > lo.sub);
        if (first >= end && advanced && end - 1 >= lo.frame) {
            const Bound tick{kInclusive, {}, end - 1, 0.0f};
            g_mem.Write(ctx, &tick, sizeof tick);
            moved_back = true;
            if (const u64 n = ++g_moved; n <= 3 || n % 10000 == 0) {
                Log("sequencer fix: update #%llu swept (%d+%g, %d+%g] within tick %d: from the tick", (unsigned long long)n,
                    lo.frame, lo.sub, hi.frame, hi.sub, end - 1);
            }
        }
    }
    if (!moved_back) {
        hle::CallOriginal(c, kFlatUpdate);
        return;
    }
    t_skip_triggers = true;
    hle::CallOriginal(c, kFlatUpdate);
    t_skip_triggers = false;
    g_mem.Write(ctx, &lo, sizeof lo); // the caller's context as it was (the instance keeps its copy until the next)
}

void TriggerImport(GuestContext* c, void* user) {
    if (t_skip_triggers) return; // (void; nothing imported)
    hle::CallOriginal(c, reinterpret_cast<u64>(user));
}

} // namespace

void Install() {
    for (const Expect& e : kExpected) {
        const Region* r = g_mem.Find(e.va);
        u32 words[4] = {};
        if (r && e.va + sizeof words <= r->base + r->size) std::memcpy(words, g_mem.Raw(e.va), sizeof words);
        if (!r || std::memcmp(words, e.words, sizeof words) != 0) {
            Log("sequencer fix: off (the game's code at 0x%llx is not 1.0.1's)", (unsigned long long)e.va);
            return;
        }
    }
    hle::OverrideGuestFunction("0x9e4f7e0", FlatUpdate);
    hle::OverrideGuestFunction("0x9f7a3b4", TriggerImport, reinterpret_cast<void*>(kTriggerImport));
    hle::OverrideGuestFunction("0x9f7a728", TriggerImport, reinterpret_cast<void*>(kTriggerImportThunk));
    Log("sequencer fix: on (repeater events every frame above 30 fps)");
}

} // namespace rt::sequencer_fix
