// Mega Snaptor's rush attack at 60 fps: no end-of-attack teleport offscreen (2026-11-15).
//
// The boss Mega Snaptor (enemy210: Act_BtlChr_218_Enemy210, AnimBP_218_enemy210) has a rush attack: four swoops
// across the field, each starting with the anim notify UnHideBone, whose handler teleports the boss to its swoop
// start, the targeted brother's starting spot + (0, 1630, 0), offscreen to the right. The notify fires once more after
// the attack has ended (AttackStandby already false). A tester saw the boss leave the screen with that attack and
// never come back when it ended the boss's turn. Traced at 60 fps (RT_BP_TRACE / RT_BP_WATCH with the boss's position
// and its home spot, CmpBattleCharaBase +0x228): the attack ends, the game's native code puts the boss home (0, 460)
// within 2 frames, then the late UnHideBone teleports it to (290, 1080) and nothing moves it again - the home spot
// stays (0, 460). Inside a turn the next attack's approach flies it home, which hides the problem; after a turn-ending
// rush there is no next attack. At 30 fps the boss comes back: the native step home comes after the teleport there.
//
// The fix: UnHideBone skips the teleport when the attack is no longer in standby. In the handler (in-memory offsets of
// ExecuteUbergraph_AnimBP_218_enemy210, 8760 bytes in memory, as a disassembly of the script shows them):
//   0x17c9  if not (K2Node_DynamicCast_bSuccess_8) jump 0x1895     (the cast of the boss to its anim interface)
//   0x1818  ... K2_SetWorldLocation(GetPlayerInitPos(TargetIndex) + (0, 1630, 0)) ...   0x1894  pop
// 0x17c9 becomes `if not (self.AttackStandby) jump 0x1894`: the same length (EX_JumpIfNot, u32 target, then an
// EX_InstanceVariable instead of the EX_LocalVariable, each with its 8-byte FProperty*), the property taken from
// 0x20bb, `if not (self.AttackStandby) jump 0x20ca`, so no name lookup is needed. The swoops (standby true) are
// unchanged; the cast never fails for this boss.
//
// How: the script's bytecode is in memory once the battle loads its class. The native USceneComponent::
// execK2_SetWorldLocation is called with the calling script's FFrame (Node +0x10, Code +0x20), so the first time the
// boss's script calls it (its first swoop), the calling function's Script array is found in the UFunction (the TArray
// holding Code, 8760 bytes long), every byte the patch relies on is checked, and the jump is rewritten (again after
// the class is loaded again; a patched script is left alone).
//
// Game 1.0.1's code (guest = main ELF + 0x8004000), checked word by word before anything is installed:
//   0xaee565c  USceneComponent::execK2_SetWorldLocation(Context, FFrame& Stack, RESULT_DECL)
#include "snaptor_fix.h"

#include <cstring>
#include <mutex>

#include "hle.h"
#include "rt.h"

namespace rt::snaptor_fix {
namespace {

constexpr u64 kExecSetWorldLocation = 0xaee565c;
constexpr u32 kExpected[4] = {0xd10383ff, 0xa90b7bfd, 0x9102c3fd, 0xa90c57f6};

constexpr u64 kFrameNode = 0x10, kFrameCode = 0x20;
constexpr u32 kScriptSize = 8760; // the script's in-memory length (0x2238)
constexpr u32 kPatchAt = 0x17c9, kSkipTo = 0x1894, kOldTarget = 0x1895;
constexpr u32 kStandbyTest = 0x20bb, kStandbyTarget = 0x20ca;
// EExprToken values (UE 4.27 Script.h)
constexpr u8 kJumpIfNot = 0x07, kLocalVariable = 0x00, kInstanceVariable = 0x01, kPopExecutionFlow = 0x4d;

std::mutex g_lock;
u64 g_reported = 0; // the last script whose differing bytes were logged (once each)

bool Read(u64 va, void* out, u64 size) {
    const Region* r = g_mem.Find(va);
    if (!r || va + size > r->base + r->size) return false;
    std::memcpy(out, g_mem.Raw(va), size);
    return true;
}

// The UFunction's Script TArray {data, num, max} that holds `code`, searched in the object's first 0x100 bytes.
bool FindScript(u64 node, u64 code, u64* data, u32* num) {
    for (u64 off = 0x28; off + 16 <= 0x100; off += 8) {
        u64 d;
        s32 n, m;
        if (!Read(node + off, &d, 8) || !Read(node + off + 8, &n, 4) || !Read(node + off + 12, &m, 4)) return false;
        if (d && n > 0 && m >= n && code >= d && code < d + static_cast<u64>(n)) {
            *data = d;
            *num = static_cast<u32>(n);
            return true;
        }
    }
    return false;
}

void TryPatch(u64 node, u64 code) {
    u64 s;
    u32 n;
    if (!FindScript(node, code, &s, &n) || n != kScriptSize) return; // another script
    u8 at[14], test[14], pop;
    u32 target, test_target;
    if (!Read(s + kPatchAt, at, sizeof at) || !Read(s + kStandbyTest, test, sizeof test) || !Read(s + kSkipTo, &pop, 1)) {
        return;
    }
    std::memcpy(&target, at + 1, 4);
    std::memcpy(&test_target, test + 1, 4);
    if (at[0] == kJumpIfNot && target == kSkipTo && at[5] == kInstanceVariable) return; // patched already
    if (at[0] != kJumpIfNot || target != kOldTarget || at[5] != kLocalVariable || test[0] != kJumpIfNot ||
        test_target != kStandbyTarget || test[5] != kInstanceVariable || pop != kPopExecutionFlow) {
        if (g_reported != s) {
            g_reported = s;
            Log("snaptor fix: a script of the boss's size whose bytes differ, not patched (0x%x: %02x %08x %02x; "
                "0x%x: %02x %08x %02x; 0x%x: %02x)",
                kPatchAt, at[0], target, at[5], kStandbyTest, test[0], test_target, test[5], kSkipTo, pop);
        }
        return;
    }
    u8 patched[14];
    patched[0] = kJumpIfNot;
    std::memcpy(patched + 1, &kSkipTo, 4);
    patched[5] = kInstanceVariable;
    std::memcpy(patched + 6, test + 6, 8); // self.AttackStandby's FProperty*
    std::memcpy(g_mem.Raw(s + kPatchAt), patched, sizeof patched);
    Log("snaptor fix: Mega Snaptor's UnHideBone patched (script at 0x%llx): no teleport after the attack has ended",
        (unsigned long long)s);
}

void SetWorldLocation(GuestContext* c, void*) {
    const u64 stack = c->x[1];
    u64 node = 0, code = 0;
    if (Read(stack + kFrameNode, &node, 8) && Read(stack + kFrameCode, &code, 8) && node) {
        // Every call looks (a few reads): a battle loaded again loads the boss's class again, unpatched.
        std::lock_guard lock(g_lock);
        TryPatch(node, code);
    }
    hle::CallOriginal(c, kExecSetWorldLocation);
}

} // namespace

void Install() {
    u32 words[4] = {};
    if (!Read(kExecSetWorldLocation, words, sizeof words) || std::memcmp(words, kExpected, sizeof words) != 0) {
        Log("snaptor fix: off (the game's code at 0x%llx is not 1.0.1's)", (unsigned long long)kExecSetWorldLocation);
        return;
    }
    hle::OverrideGuestFunction("0xaee565c", SetWorldLocation);
    Log("snaptor fix: on (Mega Snaptor stays home after its rush attack; RT_SNAPTOR_FIX=0: off)");
}

} // namespace rt::snaptor_fix
