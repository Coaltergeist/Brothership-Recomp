#include "overlay.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "options.h"
#include "perf.h"
#include "services.h"
#include "window.h"

#include "video_core/textures/decoders.h"

namespace rt::overlay {
namespace {

constexpr u32 kWidth = 1280, kHeight = 720, kBpp = 4; // RGBA8888, the size of the game's layer
constexpr u32 kBlockHeightLog2 = 4;                    // what the renderer assumes for display layers
constexpr u32 kAndroidRgba8888 = 1;

// 5x7 glyphs, one byte per row (bit 4 = leftmost column). Lower case draws as upper case.
constexpr char kGlyphChars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:/-%() ";
constexpr u8 kGlyphs[][7] = {
    {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 0 1
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}, // 2 3
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}, // 4 5
    {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, // 6 7
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}, // 8 9
    {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}, // A B
    {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}, {0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C}, // C D
    {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}, {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}, // E F
    {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}, {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, // G H
    {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}, {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}, // I J
    {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}, {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}, // K L
    {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}, {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11}, // M N
    {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}, // O P
    {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}, {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}, // Q R
    {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}, {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, // S T
    {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}, // U V
    {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}, {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}, // W X
    {0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04}, {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}, // Y Z
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}, {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00}, // . :
    {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00}, {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}, // / -
    {0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03}, {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02}, // % (
    {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08}, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // ) space
};
static_assert(sizeof kGlyphs / sizeof kGlyphs[0] == sizeof kGlyphChars - 1);

/// A linear RGBA canvas (premultiplied alpha) for the text box.
struct Canvas {
    u32 w, h;
    std::vector<u8> px;
    Canvas(u32 w_, u32 h_, u8 alpha) : w{w_}, h{h_}, px(static_cast<size_t>(w_) * h_ * kBpp) {
        for (size_t i = 3; i < px.size(); i += kBpp) px[i] = alpha; // black, partly transparent
    }
    void Text(u32 x, u32 y, u32 scale, const std::string& s) {
        for (char ch : s) {
            const char up = (ch >= 'a' && ch <= 'z') ? static_cast<char>(ch - 'a' + 'A') : ch;
            const char* at = std::strchr(kGlyphChars, up);
            const u8* g = kGlyphs[at && *at ? at - kGlyphChars : sizeof kGlyphChars - 2]; // unknown: space
            for (u32 row = 0; row < 7; ++row) {
                for (u32 col = 0; col < 5; ++col) {
                    if (!(g[row] >> (4 - col) & 1)) continue;
                    for (u32 dy = 0; dy < scale; ++dy) {
                        for (u32 dx = 0; dx < scale; ++dx) {
                            const u32 px_x = x + col * scale + dx, px_y = y + row * scale + dy;
                            if (px_x < w && px_y < h) std::memset(&px[(static_cast<size_t>(px_y) * w + px_x) * kBpp], 0xFF, kBpp);
                        }
                    }
                }
            }
            x += 6 * scale;
        }
    }
};

struct State {
    bool init = false, enabled = false;
    u64 device_address = 0;
    u64 drawn_key = ~0ULL; // perf::Frames() is not enough: redraw when the second changes (DrawKey)
};

State& S() {
    static State s;
    return s;
}

constexpr u32 kBoxX = 16, kBoxY = 16, kBoxW = 384, kBoxH = 64; // multiples of the GOB width (64 bytes)

std::atomic<unsigned> g_flag{0};       // the last flag shown (ShowFlag), 0: none
std::atomic<double> g_flag_until{0.0}; // shown until this Uptime()
constexpr double kFlagSeconds = 3.0;

/// What the box shows besides the frame rate: the flag while it is shown, else 0.
unsigned ShownFlag() {
    return Uptime() < g_flag_until.load(std::memory_order_relaxed) ? g_flag.load(std::memory_order_relaxed) : 0;
}

/// When to redraw the box: a new second measured, or the flag shown or gone.
u64 DrawKey() {
    return static_cast<u64>(Uptime()) | u64{ShownFlag()} << 32;
}

Canvas BoxCanvas(const perf::Second& sec) {
    Canvas c(kBoxW, kBoxH, 0x99);
    char line[64];
    std::snprintf(line, sizeof line, "%u FPS", sec.frames);
    c.Text(10, 8, 3, line);
    if (const unsigned flag = ShownFlag()) {
        std::snprintf(line, sizeof line, "FLAG %u", flag);
        c.Text(190, 8, 3, line);
    }
    std::snprintf(line, sizeof line, "P50 %.1f  P95 %.1f  MAX %.1f MS", sec.p50_ms, sec.p95_ms, sec.max_ms);
    c.Text(10, 40, 2, line);
    return c;
}

bool Enabled() {
    static const bool on = [] {
        const char* o = std::getenv("RT_OVERLAY"); // (wins over Debug mode)
        return o ? std::strcmp(o, "0") != 0 : window::IsOpen() && options::Debug();
    }();
    return on;
}

void Draw(u8* surface, const perf::Second& sec) {
    const Canvas c = BoxCanvas(sec);
    const size_t size = Tegra::Texture::CalculateSize(true, kBpp, kWidth, kHeight, 1, kBlockHeightLog2, 0);
    Tegra::Texture::SwizzleSubrect(std::span<u8>(surface, size), c.px, kBpp, kWidth, kHeight, 1, kBoxX, kBoxY, kBoxW,
                                   kBoxH, kBlockHeightLog2, 0, kBoxW * kBpp);
}

} // namespace

std::optional<gpu::DisplayLayer> Layer() {
    auto& s = S();
    if (!s.init) {
        s.init = true;
        s.enabled = Enabled();
        if (s.enabled) {
            const size_t size = Tegra::Texture::CalculateSize(true, kBpp, kWidth, kHeight, 1, kBlockHeightLog2, 0);
            if (size > kRuntimeRegionSize) Fatal("overlay: surface of 0x%zx bytes does not fit", size);
            // Read and written by the runtime only (Raw); the guest sees an inaccessible region.
            g_mem.Map(kRuntimeRegion, kRuntimeRegionSize, 0, kMemInaccessible, "runtime overlay");
            s.device_address = PinHostRegion(kRuntimeRegion, kRuntimeRegionSize);
            Log("overlay: on (device address 0x%llx)", (unsigned long long)s.device_address);
        }
    }
    if (!s.enabled) return std::nullopt;
    // Redrawn when a new second has been measured; the renderer may read the
    // box while it changes (a torn digit for one frame at most).
    if (const u64 key = DrawKey(); key != s.drawn_key) {
        s.drawn_key = key;
        Draw(g_mem.Raw(kRuntimeRegion), perf::LastSecond());
    }
    gpu::DisplayLayer l{};
    l.device_address = s.device_address;
    l.width = kWidth;
    l.height = kHeight;
    l.stride = kWidth;
    l.format = kAndroidRgba8888;
    l.crop_right = static_cast<s32>(kWidth);
    l.crop_bottom = static_cast<s32>(kHeight);
    l.premultiplied = true;
    return l;
}

std::optional<Box> CurrentBox() {
    if (!Enabled()) return std::nullopt;
    static u64 drawn = ~0ULL;
    static Box box{kBoxX, kBoxY, kBoxW, kBoxH, {}};
    if (const u64 key = DrawKey(); key != drawn) {
        drawn = key;
        box.rgba = BoxCanvas(perf::LastSecond()).px;
    }
    return box;
}

void ShowFlag(unsigned n) {
    g_flag.store(n, std::memory_order_relaxed);
    g_flag_until.store(Uptime() + kFlagSeconds, std::memory_order_relaxed);
}

} // namespace rt::overlay
