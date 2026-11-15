// Behaviour-tree steps at the game's 30 fps pace above 30 fps (2026-11-11).
//
// Brothership's field events are behaviour trees (an event's steps, its checks, its camera moves). Unreal moves a
// tree from one step to the next in UBehaviorTreeComponent::ProcessExecutionRequest, which TickComponent runs once a
// tick when a step was requested (bRequestedFlowUpdate) - so a chain of instant steps (a check point, a flag, an
// end) costs one frame each. The game was made at 30 fps, and some events rely on how much time those frames take.
// A tester's seesaw (Island 502a, BT_SeesawJumpSuccess_Island_502a): Mario lands, the tilt lifts Luigi over real
// time, and five steps later a check (BTS_FieldEventNearLocation, Direction_Check) wants Luigi within 6700 of the
// cliff's goal point. At 30 fps that is 167 ms after the landing: 6693.8, the cliff launch with its camera. At 60 fps
// 83 ms: 6709.9, the plain jump, the camera staying on Mario (traced with RT_BP_TRACE / RT_BP_WATCH, 2026-11-10/11).
//
// The pacing: a tree's next step waits until 1/30 s of game time has passed since its last one. Everything else in
// its tick still runs every frame (services, the running task, movement and cameras stay smooth); only the step
// change is held. How: TickComponent is reached through its vtable, so it is overridden; when a step is pending too
// early, the flag is cleared for the call (the tick runs without the step change) and set again afterwards, and the
// next tick is scheduled for the next frame as TickComponent itself does for a pending step.
//
// Game 1.0.1's code (guest = main ELF + 0x8004000), checked word by word before anything is installed:
//   0xb173df0  UBehaviorTreeComponent::TickComponent(this, DeltaTime in s0, TickType, TickFunction)
//   0xb173fa0  its test of bRequestedFlowUpdate (+0x273 bit 3) before `bl ProcessExecutionRequest` (0xb1745d8)
//   0xa5af880  UActorComponent::SetComponentTickIntervalAndCooldown(this, float)
// Fields: NextTickDeltaTime +0x284; vtable: IsComponentTickEnabled +0x378, SetComponentTickEnabled +0x368.
#include "bt_pacing.h"

#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "hle.h"
#include "rt.h"

namespace rt::bt_pacing {
namespace {

constexpr u64 kTickComponent = 0xb173df0;
constexpr u64 kFlowTest = 0xb173fa0;
constexpr u64 kSetTickInterval = 0xa5af880;
constexpr u64 kFlags = 0x273;
constexpr u8 kRequestedFlowUpdate = 0x08;
constexpr u64 kNextTickDeltaTime = 0x284;
constexpr u64 kIsComponentTickEnabled = 0x378, kSetComponentTickEnabled = 0x368;
// One step per 1/30 s; a little under it, so that two frames at 60 fps (2 x 16.7 ms, with jitter) always reach it.
constexpr float kStep = 1.0f / 30.0f - 0.002f;

struct Expect {
    u64 va;
    u32 words[4];
};
constexpr Expect kExpected[] = {
    {kTickComponent, {0xd10383ff, 0x6d0723e9, 0x4ea01c08, 0xa9087bfd}},
    {kFlowTest, {0x3949ce69, 0x371800a9, 0x2a1f03f4, 0x7100051f}},
    {kSetTickInterval, {0x9100c000, 0x14192b09, 0xbd403c00, 0xd65f03c0}},
};

std::mutex g_lock;
std::unordered_map<u64, float> g_since; // per component: game time since its last step (or since it was first seen)
std::atomic<u64> g_held{0}, g_steps{0};
std::atomic<u64> g_window_held{0}, g_window_steps{0}; // since the last summary
std::atomic<double> g_window_start{0.0};

// Every 5 s in which trees stepped: how many steps were taken and held, so a stretch of play (a battle, an event)
// shows whether the pacing was involved at all.
void Summary() {
    const double now = Uptime();
    double start = g_window_start.load();
    if (now - start < 5.0 || !g_window_start.compare_exchange_strong(start, now)) return;
    const u64 steps = g_window_steps.exchange(0), held = g_window_held.exchange(0);
    if (steps + held) {
        Log("bt pacing: %.0f-%.0f s: %llu steps taken, %llu held a frame", start, now, (unsigned long long)steps,
            (unsigned long long)held);
    }
}

void TickComponent(GuestContext* c, void*) {
    const u64 self = c->x[0];
    const float dt = hle::ArgF(c, 0);
    const u8 flags = hle::ReadGuest<u8>(self + kFlags);
    const bool pending = (flags & kRequestedFlowUpdate) != 0;
    bool hold = false;
    {
        std::lock_guard lock(g_lock);
        float& since = g_since.try_emplace(self, kStep).first->second; // (a new tree's first step is not held)
        since += dt;
        if (pending) {
            hold = since < kStep;
            if (!hold) since = 0.0f; // this tick takes the step
        }
    }
    Summary();
    if (!hold) {
        if (pending) {
            ++g_steps;
            ++g_window_steps;
        }
        hle::CallOriginal(c, kTickComponent);
        return;
    }
    hle::WriteGuest<u8>(self + kFlags, flags & ~kRequestedFlowUpdate);
    hle::CallOriginal(c, kTickComponent);
    // The step is still wanted (and a request made during this tick merges into it): set the flag again and tick
    // next frame, as TickComponent's own scheduling does while the flag is set.
    hle::WriteGuest<u8>(self + kFlags, hle::ReadGuest<u8>(self + kFlags) | kRequestedFlowUpdate);
    hle::WriteGuest<float>(self + kNextTickDeltaTime, 0.0f);
    const u64 vtable = hle::ReadGuest<u64>(self);
    if ((hle::CallGuest(c, hle::ReadGuest<u64>(vtable + kIsComponentTickEnabled), {self}) & 1) == 0) {
        hle::CallGuest(c, hle::ReadGuest<u64>(vtable + kSetComponentTickEnabled), {self, 1});
    }
    hle::CallGuestFp(c, kSetTickInterval, {self}, {0}, nullptr);
    ++g_window_held;
    if (const u64 n = ++g_held; n <= 3 || n % 10000 == 0) {
        Log("bt pacing: step #%llu of a behaviour tree held one frame (%llu steps taken so far)", (unsigned long long)n,
            (unsigned long long)g_steps.load());
    }
}

} // namespace

void Install() {
    for (const Expect& e : kExpected) {
        const Region* r = g_mem.Find(e.va);
        u32 words[4] = {};
        if (r && e.va + sizeof words <= r->base + r->size) std::memcpy(words, g_mem.Raw(e.va), sizeof words);
        if (!r || std::memcmp(words, e.words, sizeof words) != 0) {
            Log("bt pacing: off (the game's code at 0x%llx is not 1.0.1's)", (unsigned long long)e.va);
            return;
        }
    }
    hle::OverrideGuestFunction("0xb173df0", TickComponent);
    Log("bt pacing: on (behaviour-tree steps at most every 1/30 s above 30 fps; RT_BT_PACING=0: off)");
}

} // namespace rt::bt_pacing
