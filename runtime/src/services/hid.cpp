// hid: input through the 0x40000-byte shared memory block libnx reads.
//
// Offsets are libnx's HidSharedMemory layout (measured with devkitA64 offsetof).
// One handheld controller is reported connected; buttons and sticks come from the
// host's keyboard and controllers (input.h) and the input script, refreshed once
// per presented frame.
//
// Input script: RT_INPUT=<start>[~<end>]:<buttons>[:<hold>[/<period>]],...
// - start/end: a presented-frame number, seconds since start ("90s"), or a game
//   event ("@Opening.bk2": the first RomFS read of a file whose path contains
//   it). Frame numbers drift with speed (loading is paced by real time);
//   events don't. An end may also be "+<n>s": n seconds after the entry
//   started (an event's time differs between hosts; this keeps the length).
// - buttons: names joined with '+' (A, B, X, Y, L, R, ZL, ZR, Plus, Minus, Up,
//   Down, Left, Right, StickL, StickR).
// - hold: frames held (default 10); with /period the press repeats every
//   period frames until <end>.
// Example: RT_INPUT=60s~@Opening.bk2:A:10/100,@Opening.bk2:B:1000000 taps A
// every 100 frames from 60 s until the opening movie starts, then holds B.
// RT_PRESS_PLUS_AT=<frame> is short for <frame>:Plus:2.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../input.h"
#include "../ipc.h"
#include "../services.h"

namespace rt {
namespace {

constexpr u64 kSharedMemorySize = 0x40000;
constexpr u64 kNpadEntries = 0x9A00, kNpadEntrySize = 0x5000, kHandheldIndex = 8;
constexpr u64 kStyleSet = 0x0, kHandheldLifo = 0x378;
constexpr u64 kLifoTail = 0x10, kLifoCount = 0x18, kLifoStorage = 0x20, kLifoEntrySize = 0x30,
              kLifoEntries = 17;
constexpr u64 kStateButtons = 0x8, kStateStickL = 0x10, kStateStickR = 0x18, kStateAttributes = 0x20, kStateSize = 0x28;
constexpr u32 kStyleHandheld = 1u << 1, kAttributeIsConnected = 1u << 0;
/// HidNpadButton bits (libnx).
constexpr struct {
    const char* name;
    u64 bit;
} kButtons[] = {{"A", 1u << 0},      {"B", 1u << 1},       {"X", 1u << 2},     {"Y", 1u << 3},
                {"StickL", 1u << 4}, {"StickR", 1u << 5},  {"L", 1u << 6},     {"R", 1u << 7},
                {"ZL", 1u << 8},     {"ZR", 1u << 9},      {"Plus", 1u << 10}, {"Minus", 1u << 11},
                {"Left", 1u << 12},  {"Up", 1u << 13},     {"Right", 1u << 14}, {"Down", 1u << 15}};

/// When a script entry starts or ends.
struct Trigger {
    enum Kind { kNever, kFrame, kSeconds, kEvent, kAfterStart } kind = kNever;
    u64 value = 0;    // frame or seconds (kAfterStart: after the entry's start)
    std::string name; // event
    s64 fired = -1;   // the frame it happened at
};

struct Press {
    Trigger start, end;
    u64 hold, period, buttons; // period 0: once
    std::string text;
    std::chrono::steady_clock::time_point started{};
};

struct HidState {
    std::shared_ptr<KSharedMemory> shmem;
    u64 sampling = 0, tail = 0, frame = 0;
    std::vector<Press> script;
    std::map<std::string, u64> events; // first frame each event was seen at
};

u64 ParseButtons(const std::string& names, const std::string& entry) {
    u64 bits = 0;
    size_t at = 0;
    while (at <= names.size()) {
        const size_t end = std::min(names.find('+', at), names.size());
        const std::string name = names.substr(at, end - at);
        u64 bit = 0;
        for (const auto& b : kButtons) {
            if (name == b.name) bit = b.bit;
        }
        if (!bit) Fatal("RT_INPUT: unknown button \"%s\" in \"%s\"", name.c_str(), entry.c_str());
        bits |= bit;
        at = end + 1;
    }
    return bits;
}

Trigger ParseTrigger(const std::string& t, const std::string& entry) {
    Trigger trig;
    if (t.empty()) Fatal("RT_INPUT: empty start or end in \"%s\"", entry.c_str());
    if (t[0] == '+') {
        if (t.size() < 3 || t.back() != 's') Fatal("RT_INPUT: a relative end is +<n>s, not \"%s\" in \"%s\"", t.c_str(), entry.c_str());
        trig.kind = Trigger::kAfterStart;
        trig.value = std::stoull(t.substr(1, t.size() - 2));
    } else if (t[0] == '@') {
        trig.kind = Trigger::kEvent;
        trig.name = t.substr(1);
    } else if (t.back() == 's') {
        trig.kind = Trigger::kSeconds;
        trig.value = std::stoull(t.substr(0, t.size() - 1));
    } else {
        trig.kind = Trigger::kFrame;
        trig.value = std::stoull(t);
    }
    return trig;
}

std::vector<Press> ParseScript() {
    std::vector<Press> script;
    if (const char* e = std::getenv("RT_PRESS_PLUS_AT")) {
        const std::string t = std::string(e) + ":Plus:2";
        script.push_back({ParseTrigger(e, t), {}, 2, 0, 1u << 10, t});
    }
    const char* e = std::getenv("RT_INPUT");
    const std::string text = e ? e : "";
    for (size_t at = 0; at < text.size();) {
        const size_t end = std::min(text.find(',', at), text.size());
        const std::string entry = text.substr(at, end - at);
        const size_t c1 = entry.find(':'), c2 = entry.find(':', c1 + 1);
        if (c1 == std::string::npos) {
            Fatal("RT_INPUT: \"%s\" is not <start>[~<end>]:<buttons>[:<hold>[/<period>]]", entry.c_str());
        }
        Press p{};
        const std::string when = entry.substr(0, c1);
        const size_t tilde = when.find('~');
        p.start = ParseTrigger(when.substr(0, tilde), entry);
        if (p.start.kind == Trigger::kAfterStart) Fatal("RT_INPUT: \"%s\" starts relative to itself", entry.c_str());
        if (tilde != std::string::npos) p.end = ParseTrigger(when.substr(tilde + 1), entry);
        p.buttons = ParseButtons(entry.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1), entry);
        p.hold = 10;
        if (c2 != std::string::npos) {
            const std::string h = entry.substr(c2 + 1);
            const size_t slash = h.find('/');
            p.hold = std::stoull(h.substr(0, slash));
            if (slash != std::string::npos) p.period = std::stoull(h.substr(slash + 1));
        }
        p.text = entry;
        script.push_back(p);
        at = end + 1;
    }
    return script;
}

HidState& State() {
    static HidState s = [] {
        HidState h;
        h.script = ParseScript();
        return h;
    }();
    return s;
}

// The other input LIFOs (libnx offsets): one empty sample each per frame. A LIFO
// with no samples is never seen on hardware; UE4 then reads stale state (it
// stopped on "touch ID [25] bigger than allowed [16]").
constexpr u64 kTouchLifo = 0x400, kTouchEntrySize = 0x298;
constexpr u64 kMouseLifo = 0x3400, kMouseEntrySize = 0x30;
constexpr u64 kKeyboardLifo = 0x3800, kKeyboardEntrySize = 0x38;

/// Writes the next sample of a 17-entry LIFO: the atomic storage's sampling
/// number, then the state, which starts with its own sampling number. All LIFOs
/// advance together (one sample per frame), so they share tail and sampling.
void PushSample(u64 lifo, u64 entry_size, const u8* state_rest = nullptr, size_t rest_size = 0) {
    auto& s = State();
    std::vector<u8> entry(entry_size);
    std::memcpy(entry.data(), &s.sampling, 8);     // atomic storage sampling number
    std::memcpy(entry.data() + 8, &s.sampling, 8); // state.sampling_number
    if (rest_size) std::memcpy(entry.data() + 16, state_rest, rest_size);
    s.shmem->Write(lifo + kLifoStorage + s.tail * entry_size, entry.data(), entry.size());
    const u64 header[3] = {kLifoEntries, s.tail, 1}; // buffer_count, tail, count
    s.shmem->Write(lifo + 8, header, sizeof header);
}

/// The stick-direction button bits the system derives from a stick (HidNpadButton StickLLeft = 1 << 16 ...
/// StickRDown = 1 << 23), at half deflection.
u64 StickButtons(const input::Pad& p) {
    constexpr s32 kHalf = 0x4000;
    const auto dirs = [](s32 x, s32 y, u32 shift) {
        const u32 bits = (x < -kHalf) | (y > kHalf) << 1 | (x > kHalf) << 2 | (y < -kHalf) << 3;
        return u64{bits} << shift;
    };
    return dirs(p.lx, p.ly, 16) | dirs(p.rx, p.ry, 20);
}

void WriteHandheld(u64 buttons, const input::Pad& sticks = {}) {
    auto& s = State();
    if (!s.shmem) {
        return;
    }
    const u64 npad = kNpadEntries + kHandheldIndex * kNpadEntrySize;
    const u32 style = kStyleHandheld;
    s.shmem->Write(npad + kStyleSet, &style, 4);

    s.tail = (s.tail + 1) % kLifoEntries;
    ++s.sampling;
    u8 rest[kStateSize - 8] = {};
    std::memcpy(rest + kStateButtons - 8, &buttons, 8);
    const s32 stick_words[4] = {sticks.lx, sticks.ly, sticks.rx, sticks.ry};
    std::memcpy(rest + kStateStickL - 8, stick_words, 8);
    std::memcpy(rest + kStateStickR - 8, stick_words + 2, 8);
    std::memcpy(rest + kStateAttributes - 8, &kAttributeIsConnected, 4);
    static_assert(8 + kStateSize <= kLifoEntrySize);
    PushSample(npad + kHandheldLifo, kLifoEntrySize, rest, sizeof rest);
    PushSample(kTouchLifo, kTouchEntrySize); // count 0: no touches
    PushSample(kMouseLifo, kMouseEntrySize);
    PushSample(kKeyboardLifo, kKeyboardEntrySize);
}

class AppletResource : public Service {
public:
    AppletResource() : Service("hid:IAppletResource") {
        Register(0, "GetSharedMemoryHandle", [](IpcContext& ctx) {
            auto& s = State();
            if (!s.shmem) {
                s.shmem = std::make_shared<KSharedMemory>(kSharedMemorySize);
                s.shmem->name = "hid";
                WriteHandheld(0);
            }
            ctx.out_copy.push_back(g_handles.Add(s.shmem));
        });
    }
};

class ActiveVibrationDeviceList : public Service {
public:
    ActiveVibrationDeviceList() : Service("hid:IActiveVibrationDeviceList") {
        Stub(0, "ActivateVibrationDevice");
    }
};

class Hid : public Service {
public:
    Hid() : Service("hid") {
        Register(0, "CreateAppletResource", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<AppletResource>()); });
        Stub(1, "ActivateDebugPad");
        Stub(11, "ActivateTouchScreen");
        Stub(21, "ActivateMouse");
        Stub(31, "ActivateKeyboard");
        // Six-axis sensor: accepted; the shared-memory states stay zero (no motion).
        Stub(66, "StartSixAxisSensor");
        Stub(67, "StopSixAxisSensor");
        Stub(68, "IsSixAxisSensorFusionEnabled", 1);
        Stub(69, "EnableSixAxisSensorFusion");
        Stub(79, "SetGyroscopeZeroDriftMode");
        Stub(82, "IsSixAxisSensorAtRest", 1);
        Stub(100, "SetSupportedNpadStyleSet");
        Register(101, "GetSupportedNpadStyleSet", [](IpcContext& ctx) { ctx.Push<u32>(kStyleHandheld); });
        Stub(102, "SetSupportedNpadIdType");
        Stub(103, "ActivateNpad");
        Stub(104, "DeactivateNpad");
        Stub(109, "ActivateNpadWithRevision");
        Stub(120, "SetNpadJoyHoldType");
        Register(121, "GetNpadJoyHoldType", [](IpcContext& ctx) { ctx.Push<u64>(1); });
        // Vibration: one linear resonant actuator per side, as drippu reports for a
        // handheld controller. Values sent are dropped.
        Register(200, "GetVibrationDeviceInfo", [](IpcContext& ctx) {
            const u32 handle = ctx.Pop<u32>(); // npad_type, npad_id, device_index, -
            const u32 kLinearResonantActuator = 1, position = (handle >> 16 & 0xFF) == 0 ? 1 : 2;
            ctx.Push<u32>(kLinearResonantActuator);
            ctx.Push<u32>(position); // 1 left, 2 right
        });
        Stub(201, "SendVibrationValue");
        Stub(202, "GetActualVibrationValue", 0x10);
        Register(203, "CreateActiveVibrationDeviceList",
                 [](IpcContext& ctx) { ctx.PushObject(std::make_shared<ActiveVibrationDeviceList>()); });
        Stub(204, "PermitVibration");
        Register(205, "IsVibrationPermitted", [](IpcContext& ctx) { ctx.Push<u8>(1); });
        Stub(206, "SendVibrationValues");
        Register(211, "IsVibrationDeviceMounted", [](IpcContext& ctx) { ctx.Push<u8>(1); });
    }
};

const bool registered = [] {
    RegisterServiceFactory("hid", [] { return std::make_shared<Hid>(); });
    return true;
}();

} // namespace

namespace {

const auto g_start_time = std::chrono::steady_clock::now();

/// Marks a trigger as fired at `frame` once its condition holds.
bool Check(Trigger& t, u64 frame, const HidState& s) {
    if (t.fired >= 0) return true;
    bool now = false;
    switch (t.kind) {
    case Trigger::kNever:
    case Trigger::kAfterStart: return false; // HidOnFrame
    case Trigger::kFrame: now = frame >= t.value; break;
    case Trigger::kSeconds:
        now = std::chrono::steady_clock::now() - g_start_time >= std::chrono::seconds(t.value);
        break;
    case Trigger::kEvent:
        for (const auto& [name, at] : s.events) {
            if (name.find(t.name) != std::string::npos) now = true;
        }
        break;
    }
    if (now) t.fired = static_cast<s64>(frame);
    return now;
}

} // namespace

void InputEvent(const std::string& name) {
    auto& s = State();
    if (s.events.emplace(name, s.frame).second && !s.script.empty()) {
        Log("hid: event %s at frame %llu (t=%.1f s)", name.c_str(), (unsigned long long)s.frame, Uptime());
    }
}

void HidOnFrame(u64 frame) {
    auto& s = State();
    s.frame = frame;
    u64 buttons = 0;
    for (auto& p : s.script) {
        const bool was_started = p.start.fired >= 0, was_ended = p.end.fired >= 0;
        if (!Check(p.start, frame, s)) continue;
        if (!was_started) {
            Log("hid: input script %s starts at frame %llu (t=%.1f s)", p.text.c_str(), (unsigned long long)frame, Uptime());
            p.started = std::chrono::steady_clock::now();
        }
        if (p.end.kind == Trigger::kAfterStart && p.end.fired < 0 &&
            std::chrono::steady_clock::now() - p.started >= std::chrono::seconds(p.end.value)) {
            p.end.fired = static_cast<s64>(frame);
        }
        if (Check(p.end, frame, s)) {
            if (!was_ended) Log("hid: input script %s ends at frame %llu (t=%.1f s)", p.text.c_str(), (unsigned long long)frame, Uptime());
            continue;
        }
        const u64 since = frame - static_cast<u64>(p.start.fired);
        if (p.period ? since % p.period < p.hold : since < p.hold) buttons |= p.buttons;
    }
    const input::Pad host = input::Current();
    WriteHandheld(buttons | host.buttons | StickButtons(host), host);
}

} // namespace rt
