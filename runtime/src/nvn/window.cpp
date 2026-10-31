// NVNwindowBuilder, NVNwindow, acquire and present (NVN_PLAN 9.1, 9.2).
//
// The layer keeps the game's window to itself: the game still creates its
// `vi` layer and passes it as the native window, but nothing is sent there.
//
// Builder: inline in the game's storage, in the layer's layout (40 bytes; SDK
// 3.5.0 writes up to +36, SDK 12.3.2 up to +48 in a larger builder). As NVN
// (BRO 0x5785a0-0x578650, XC2 0x3ca6b8-0x3ca6fc): SetDefaults clears all but
// the device; the present interval is kept minus one, so it reads back as 1
// after SetDefaults; GetTexture(i) gives 0 unless i < the texture count.
//
// Display model (9.2; the console's own buffer queue is being studied in
// sdk-rea, INVESTIGATION_nvn_window.md): textures are handed out in turn
// (0, 1, 2, 0 ... over the active ones); a presented texture is on screen
// from the first 60 Hz vsync at least `interval` vsyncs after the previous
// frame's (interval 0: at once), and stays until the next frame replaces it.
// nvnWindowAcquireTexture blocks the calling thread until its texture has
// left the screen and the GPU work presented with it is done, as the
// console's dequeue does; its sync is then already signaled. Each present
// counts as a frame (perf.h) and refreshes input (HidOnFrame). RT_CAPTURE=<frame>[,<frame>...] (as vi's)
// writes frame_<n>.png of the presented texture (RGBA8 / BGRA8 formats): read back by the GPU without
// waiting, picked up at a later present, encoded on the capture thread. A capture still being read back
// when the game stops presenting is not written. F9 in the window: the next three presented frames are
// captured too and their draws logged ("nvn drawlog" lines, draws.cpp), for a glitch seen in play.
// RT_TRACE_PRESENT=1 (as vi's) logs each acquire (time asleep until the texture leaves the screen, time
// waiting for its GPU work) and each present (texture, interval, display time).
// The presented texture is shown in the host window (display.cpp). Not yet: crops.
#include "nvn.h"
#include "display.h"
#include "images.h"
#include "render.h"
#include "texture.h"

#include "../input.h"
#include "../options.h"
#include "../overlay.h"

#include "../perf.h"
#include "../services.h"
#include "../threads.h"
#include "../platform/platform.h"

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace rt::nvn {

namespace {

using Clock = std::chrono::steady_clock;
constexpr auto kVsync = std::chrono::nanoseconds(1'000'000'000 / 60);

bool TracePresent() {
    static const bool trace = std::getenv("RT_TRACE_PRESENT") != nullptr;
    return trace;
}

// RT_FPS=30|60|uncapped: the frame-rate cap, whatever the game asks (present interval 2, 1 or 0); F8 in
// the window cycles through the three. It only caps: the game's own limiter still holds Brothership at
// 30 fps without the 60 fps mod (which asks for interval 0). -1: not set, the game's interval.
std::atomic<s32> g_fps_interval{[] {
    const char* v = options::Get("RT_FPS");
    if (!v) return -1;
    const std::string s = v;
    const s32 n = s == "30" ? 2 : s == "60" ? 1 : s == "uncapped" ? 0 : -2;
    if (n == -2) Fatal("RT_FPS=%s: expected 30, 60 or uncapped", v);
    return n;
}()};

const char* FpsName(s32 interval) { return interval == 2 ? "30" : interval == 1 ? "60" : "uncapped"; }

// RT_PRESENT_INTERVAL=<n>: at most n vsyncs per frame, whatever the game asks (1: up to 60 fps, 0: no
// waiting), as an emulator's "vsync off"; the game still reads back its own setting. Unset: the game's.
// RT_FPS wins over it.
s32 PacedInterval(s32 interval) {
    static const s32 cap = [] {
        if (const s32 f = g_fps_interval; f >= 0) Log("nvn window: frame rate %s (RT_FPS)", FpsName(f));
        const char* v = std::getenv("RT_PRESENT_INTERVAL");
        if (!v) return -1;
        const s32 n = std::atoi(v);
        if (n < 0) Fatal("RT_PRESENT_INTERVAL=%s: expected 0 or more", v);
        Log("nvn window: present interval at most %d (RT_PRESENT_INTERVAL)", n);
        return n;
    }();
    if (const s32 f = g_fps_interval; f >= 0) return f;
    return cap >= 0 && interval > cap ? cap : interval;
}

/// F8: the next frame rate after the one in effect (30 -> 60 -> uncapped -> 30).
void CycleFps(s32 effective) {
    const s32 next = effective >= 2 ? 1 : effective == 1 ? 0 : 2;
    g_fps_interval = next;
    Log("nvn window: F8 (t=%.1f s): frame rate %s", Uptime(), FpsName(next));
}

struct WindowBuilder {
    u64 device;
    u64 textures; // NVNtexture* const*
    u64 native;
    s32 num_textures, num_active;
    s32 interval_minus_one;
    s32 pad;
};
static_assert(sizeof(WindowBuilder) == 40);

template <typename F>
void Change(u64 builder, F&& f) {
    auto b = Load<WindowBuilder>(builder);
    f(b);
    Store(builder, b);
}

struct Window : Object {
    static constexpr Type kType = Type::kWindow;
    std::mutex mu; // everything below
    u64 device = 0, native = 0;
    std::vector<u64> textures; // the game's NVNtexture objects
    s32 num_active = 0;        // 0: all
    s32 interval = 1;
    u32 next = 0; // the next texture to hand out
    struct Slot {
        Clock::time_point released{}; // off the screen (time_point::max(): still on it)
        u64 queue = 0, work = 0;       // the GPU work presented with it
        bool acquired = false;
    };
    std::vector<Slot> slots;
    Clock::time_point epoch{}, last_display{}; // the vsync grid; the latest frame's display time
    s32 on_screen = -1;
    bool any = false;
    u64 frames = 0; // presents so far (input refresh, HidOnFrame)

    u32 Active() const {
        return num_active > 0 && num_active <= static_cast<s32>(textures.size()) ? num_active
                                                                                    : static_cast<u32>(textures.size());
    }
};

const std::set<u64>& CaptureFrames() {
    static const std::set<u64> frames = [] {
        std::set<u64> f;
        std::istringstream in(std::getenv("RT_CAPTURE") ? std::getenv("RT_CAPTURE") : "");
        for (std::string x; std::getline(in, x, ',');) {
            if (!x.empty()) f.insert(std::stoull(x));
        }
        return f;
    }();
    return frames;
}

// Captures in flight: each presented texture's read-back (StartDownload) is checked at every present,
// without waiting, on the presenting thread (which may use the allocator); the pixels then go to the
// capture thread for the PNG (capture.cpp).
struct PendingCapture {
    u64 frame;
    u32 width, height;
    bool rgba; // otherwise BGRA
    std::unique_ptr<ImageDownload> download;
};
std::mutex g_captures_mu;
std::vector<PendingCapture> g_captures;

/// Starts the read-back of texture `texture` (level 0) for frame_<frame>.png.
void Capture(u64 frame, u64 texture) {
    const TextureState t = TextureStateOf(texture);
    const u32 f = t.settings.format;
    const bool rgba = f == 0x25 || f == 0x38 || f == 0x37 || f == 0x6d, bgra = f == 0x9a || f == 0x9c || f == 0x99 || f == 0x9b;
    if (!rgba && !bgra) {
        Log("nvn: frame %llu: no capture of format 0x%x", (unsigned long long)frame, f);
        return;
    }
    NoteImageUse(t, ImageUse::kCapture);
    std::unique_ptr<ImageDownload> d = StartDownload(t, 0);
    if (!d || d->size < u64{4} * d->width * d->height) { // (a scaled image: its own size, RESOLUTION_PLAN Phase 3)
        Log("nvn: frame %llu: the presented texture has no image yet", (unsigned long long)frame);
        return;
    }
    Log("nvn: frame %llu (t=%.1f s): capturing to frame_%llu.png", (unsigned long long)frame, Uptime(),
        (unsigned long long)frame);
    std::lock_guard lock(g_captures_mu);
    const u32 width = d->width, height = d->height;
    g_captures.push_back({frame, width, height, rgba, std::move(d)});
}

/// Hands the finished read-backs to the capture thread.
void PollCaptures() {
    std::lock_guard lock(g_captures_mu);
    for (auto it = g_captures.begin(); it != g_captures.end();) {
        if (!it->download->Ready()) {
            ++it;
            continue;
        }
        CaptureAsync([px = it->download->Take(), frame = it->frame, w = it->width, h = it->height, rgba = it->rgba]() mutable {
            if (rgba) {
                for (size_t i = 0; i + 3 < px.size(); i += 4) std::swap(px[i], px[i + 2]);
            }
            const std::string path = "frame_" + std::to_string(frame) + ".png";
            if (!WriteBgraPng(path, w, h, px.data())) Log("nvn: could not write %s", path.c_str());
        });
        it = g_captures.erase(it); // the read-back's buffer is freed here, on this thread
    }
}

} // namespace

// --- WindowBuilder ------------------------------------------------------------------------------

NVN_PROC(nvnWindowBuilderSetDevice, void, u64 builder, u64 device) {
    Change(builder, [&](WindowBuilder& b) { b.device = device; });
}
NVN_PROC(nvnWindowBuilderSetDefaults, void, u64 builder) {
    Change(builder, [](WindowBuilder& b) {
        const u64 device = b.device;
        b = WindowBuilder{};
        b.device = device;
    });
}
NVN_PROC(nvnWindowBuilderSetNativeWindow, void, u64 builder, u64 native) {
    Change(builder, [&](WindowBuilder& b) { b.native = native; });
}
NVN_PROC(nvnWindowBuilderSetTextures, void, u64 builder, s32 count, u64 textures) {
    Change(builder, [&](WindowBuilder& b) {
        b.num_textures = count;
        b.textures = textures;
    });
}
// Interval settings are logged with the guest's return address (who paces the game: 2 = 30 fps).
NVN_PROC(nvnWindowBuilderSetPresentInterval, void, u64 builder, s32 interval) {
    Log("nvn window: builder present interval %d (guest return address 0x%llx)", interval,
        (unsigned long long)c->x[30]);
    Change(builder, [&](WindowBuilder& b) { b.interval_minus_one = interval - 1; });
}
NVN_PROC(nvnWindowBuilderSetNumActiveTextures, void, u64 builder, s32 count) {
    Change(builder, [&](WindowBuilder& b) { b.num_active = count; });
}
NVN_PROC(nvnWindowBuilderGetDevice, u64, u64 builder) { return Load<WindowBuilder>(builder).device; }
NVN_PROC(nvnWindowBuilderGetNumTextures, s32, u64 builder) { return Load<WindowBuilder>(builder).num_textures; }
NVN_PROC(nvnWindowBuilderGetTexture, u64, u64 builder, s32 i) {
    const auto b = Load<WindowBuilder>(builder);
    return b.num_textures > i ? hle::ReadGuest<u64>(b.textures + 8 * static_cast<s64>(i)) : 0;
}
NVN_PROC(nvnWindowBuilderGetPresentInterval, s32, u64 builder) {
    return Load<WindowBuilder>(builder).interval_minus_one + 1;
}
NVN_PROC(nvnWindowBuilderGetNumActiveTextures, s32, u64 builder) { return Load<WindowBuilder>(builder).num_active; }
NVN_PROC(nvnWindowBuilderGetNativeWindow, u64, u64 builder) { return Load<WindowBuilder>(builder).native; }

// --- Window ---------------------------------------------------------------------------------------

NVN_PROC(nvnWindowInitialize, bool, u64 window, u64 builder) {
    const auto b = Load<WindowBuilder>(builder);
    if (b.num_textures < 2 || b.num_textures > 16) {
        Fatal("nvn: nvnWindowInitialize with %d textures (the layer handles 2 to 16)", b.num_textures);
    }
    Window* w = Create<Window>(window);
    w->device = b.device;
    w->native = b.native;
    for (s32 i = 0; i < b.num_textures; ++i) {
        const u64 t = hle::ReadGuest<u64>(b.textures + 8ULL * i);
        TextureStateOf(t); // stops the run for an object that is no initialised texture
        w->textures.push_back(t);
    }
    w->num_active = b.num_active;
    w->interval = b.interval_minus_one + 1;
    w->slots.resize(w->textures.size());
    return true;
}
NVN_PROC(nvnWindowFinalize, void, u64 window) { Destroy<Window>(window); }
NVN_PROC(nvnWindowSetDebugLabel, void, u64 window, u64 label) { Get<Window>(window); }
NVN_PROC(nvnWindowGetNativeWindow, u64, u64 window) { return Get<Window>(window)->native; }
NVN_PROC(nvnWindowGetNumTextures, s32, u64 window) {
    Window* w = Get<Window>(window);
    std::lock_guard lock(w->mu);
    return static_cast<s32>(w->textures.size());
}
NVN_PROC(nvnWindowGetPresentInterval, s32, u64 window) {
    Window* w = Get<Window>(window);
    std::lock_guard lock(w->mu);
    return w->interval;
}
NVN_PROC(nvnWindowSetPresentInterval, void, u64 window, s32 interval) {
    Window* w = Get<Window>(window);
    if (interval < 0) Fatal("nvn: nvnWindowSetPresentInterval(%d)", interval);
    std::lock_guard lock(w->mu);
    if (interval != w->interval) {
        Log("nvn window: present interval %d -> %d (guest return address 0x%llx)", w->interval, interval,
            (unsigned long long)c->x[30]);
    }
    w->interval = interval; // from the next present
}

// NVNwindowAcquireTextureResult: 0 = success.
NVN_PROC(nvnWindowAcquireTexture, u32, u64 window, u64 sync, u64 index_out) {
    Window* w = Get<Window>(window);
    u32 i;
    Window::Slot slot;
    {
        std::lock_guard lock(w->mu);
        i = w->next % w->Active();
        w->next = i + 1;
        if (w->slots[i].acquired) Fatal("nvn: nvnWindowAcquireTexture: texture %u is acquired and not presented", i);
        if (w->slots[i].released == Clock::time_point::max()) {
            Fatal("nvn: nvnWindowAcquireTexture: texture %u is still on the screen (one active texture?)", i);
        }
        w->slots[i].acquired = true;
        slot = w->slots[i];
    }
    const auto t0 = Clock::now();
    RunHostWait(kHostWaitPresent, [&] { platform::SleepUntil(slot.released); });
    const auto t1 = Clock::now();
    if (slot.queue) WaitQueue(slot.queue, slot.work);
    if (TracePresent()) {
        const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        Log("nvn window: acquire %u (t=%.3f s): %.1f ms until off screen, %.1f ms for its GPU work", i, Uptime(),
            ms(t1 - t0), ms(Clock::now() - t1));
    }
    SignalSyncOnHost(sync);
    hle::WriteGuest<s32>(index_out, static_cast<s32>(i));
    return 0;
}

NVN_PROC(nvnQueuePresentTexture, void, u64 queue, u64 window, s32 index) {
    Window* w = Get<Window>(window);
    const u64 work = QueueSubmitted(queue);
    u64 frame;
    {
        std::lock_guard lock(w->mu);
        if (index < 0 || index >= static_cast<s32>(w->slots.size()) || !w->slots[index].acquired) {
            Fatal("nvn: nvnQueuePresentTexture of texture %d, which is not acquired", index);
        }
        if (input::TakeFpsKey()) CycleFps(PacedInterval(w->interval));
        const auto now = Clock::now();
        if (!w->any) w->epoch = w->last_display = now, w->any = true;
        Clock::time_point display = now;
        if (const s32 interval = PacedInterval(w->interval); interval > 0) {
            const auto earliest = std::max(now, w->last_display + interval * kVsync);
            const auto vsyncs = (earliest - w->epoch + kVsync - std::chrono::nanoseconds(1)) / kVsync;
            display = w->epoch + vsyncs * kVsync;
        }
        if (w->on_screen >= 0) w->slots[w->on_screen].released = display;
        Window::Slot& s = w->slots[index];
        s.acquired = false;
        s.released = Clock::time_point::max();
        s.queue = queue;
        s.work = work;
        w->on_screen = index;
        w->last_display = display;
        frame = ++w->frames;
        if (TracePresent()) {
            const QueueTimes qt = TakeQueueTimes();
            Log("nvn window: present %d (t=%.3f s), interval %d, on screen in %.1f ms; since the last present: "
                "%llu submissions translated in %.1f ms, %.1f ms waiting for the GPU, GPU latency up to %.1f ms; "
                "this thread: %s",
                index, Uptime(), w->interval, std::chrono::duration<double, std::milli>(display - now).count(),
                (unsigned long long)qt.submits, qt.submit_ms, qt.wait_ms, qt.gpu_max_ms, TakeSvcTimes().c_str());
        }
    }
    perf::OnPresent();
    TraceWaitsPresent(frame);
    const u64 shown = w->textures[static_cast<size_t>(index)];
    // F9: draws translated after this present belong to the next frame; log and capture three such frames
    static u64 debug_last = 0; // the last frame F9 captures, 0: none
    const bool debug = debug_last && frame <= debug_last;
    // RT_NVN_F9_AT=<frame>: F9 by itself at that frame (headless runs)
    static const u64 f9_at = std::getenv("RT_NVN_F9_AT") ? std::strtoull(std::getenv("RT_NVN_F9_AT"), nullptr, 0) : 0;
    if ((input::TakeCaptureKey() || (f9_at && frame == f9_at)) && !debug) {
        debug_last = frame + 3;
        SetDrawLogFrame(frame + 1);
        Log("nvn: F9 (t=%.1f s): logging the draws of frames %llu-%llu and capturing them", Uptime(),
            (unsigned long long)frame + 1, (unsigned long long)debug_last);
    } else if (debug) {
        SetDrawLogFrame(frame < debug_last ? frame + 1 : 0);
    }
    if (input::TakeToggleKey()) {
        Log("nvn: F10 (t=%.1f s, frame %llu): the RT_NVN_TOGGLE_PROGRAMS dispatches now %s", Uptime(),
            (unsigned long long)frame, FlipToggledPrograms() ? "skipped" : "run");
    }
    // F7 / a controller's Capture button: the player's flag for something seen in play (a hitch, a glitch):
    // a numbered log line to find the moment by, and a capture of this frame
    static unsigned flags = 0;
    static u64 flag_poll_until = 0; // keep polling its capture until this frame
    const bool flag = input::TakeFlagKey();
    if (flag) {
        ++flags;
        flag_poll_until = frame + 120;
        Log("FLAG %u (t=%.1f s, frame %llu): flagged by the player; see the lines before this one", flags, Uptime(),
            (unsigned long long)frame);
        overlay::ShowFlag(flags);
    }
    if (CaptureFrames().count(frame) || debug || flag) Capture(frame, shown);
    if (!CaptureFrames().empty() || debug_last || frame <= flag_poll_until) PollCaptures();
    ShowInWindow(TextureStateOf(shown));
    HidOnFrame(frame); // as vi's present does: input and the input script advance per frame
    SavePipelineCacheIfChanged();
    PipelineFrameStart();
}

} // namespace rt::nvn
