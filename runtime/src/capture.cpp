// Frame capture: block-linear (16Bx2 GOBs) -> linear RGB -> PNG.
//
// The deswizzle is the inverse of libnx framebuffer.c's _convertToBlocklinear:
// blocks of (1 << block_height_log2) GOBs, each GOB 512 bytes = 64 bytes x 8
// rows, stored as 32 16-byte units in swizzled order.
//
// Captures are written on background threads (a few, so that capturing every
// frame keeps up: one PNG encode takes longer than a 30 fps frame) (CaptureAsync): where a frame is taken only its pixels are copied out, and
// the deswizzle, the conversion and the PNG encode happen there, so a capture
// costs the game (or the render thread) no encode.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include <SDL3/SDL_surface.h>

#include "services.h"

namespace rt {
namespace {

// RGB8 rows -> PNG, written by SDL (its PNG writer deflates; no SDL_Init needed for surfaces).
bool WritePng(const std::string& path, u32 w, u32 h, const std::vector<u8>& rgb) {
    SDL_Surface* s = SDL_CreateSurfaceFrom(static_cast<int>(w), static_cast<int>(h), SDL_PIXELFORMAT_RGB24,
                                           const_cast<u8*>(rgb.data()), static_cast<int>(w * 3));
    if (!s) return false;
    const bool ok = SDL_SavePNG(s, path.c_str());
    SDL_DestroySurface(s);
    return ok;
}

// The capture thread's queue. Never destroyed: the runtime leaves by _Exit (after FinishCaptures),
// and a capture may be queued while static destructors run.
struct Writer {
    std::mutex mu;
    std::condition_variable work, idle;
    std::deque<std::function<void()>> jobs;
    u32 busy = 0;        // jobs being written
    bool started = false;
};
Writer& W() {
    static Writer* w = new Writer;
    return *w;
}
constexpr size_t kMaxQueued = 48; // each holds a frame's pixels (~3.7 MB at 1280x720)
/// Writer threads: a quarter of the host's, 2 to 6.
u32 Writers() {
    return std::clamp(std::thread::hardware_concurrency() / 4, 2u, 6u);
}

void WriterMain() {
    Writer& w = W();
    std::unique_lock lock(w.mu);
    for (;;) {
        w.work.wait(lock, [&] { return !w.jobs.empty(); });
        std::function<void()> job = std::move(w.jobs.front());
        w.jobs.pop_front();
        ++w.busy;
        lock.unlock();
        job();
        lock.lock();
        --w.busy;
        if (w.jobs.empty() && !w.busy) w.idle.notify_all();
    }
}

bool DeswizzleToPng(const PresentedBuffer& buf, const std::vector<u8>& tiled, const std::string& path) {
    const u32 bpp = buf.format == 4 ? 2 : 4; // RGB565 or 32-bit RGBA/RGBX/BGRA
    const u32 gobs_per_block = 1u << buf.block_height_log2, block_px = 8u << buf.block_height_log2;
    const u32 width_blocks = buf.pitch / 64;
    const u32 height_blocks = (buf.height + block_px - 1) / block_px;

    std::vector<u8> linear(static_cast<size_t>(buf.pitch) * height_blocks * block_px);
    const u8* gob = tiled.data();
    for (u32 by = 0; by < height_blocks; ++by) {
        for (u32 bx = 0; bx < width_blocks; ++bx) {
            for (u32 gy = 0; gy < gobs_per_block; ++gy, gob += 512) {
                const u32 y0 = by * block_px + gy * 8;
                for (u32 i = 0; i < 32; ++i) {
                    const u32 y = ((i >> 1) & 0x06) | (i & 0x01);
                    const u32 x = ((i << 3) & 0x10) | ((i << 1) & 0x20);
                    std::memcpy(&linear[static_cast<size_t>(y0 + y) * buf.pitch + bx * 64 + x], gob + i * 16, 16);
                }
            }
        }
    }

    std::vector<u8> rgb(static_cast<size_t>(buf.width) * buf.height * 3);
    for (u32 y = 0; y < buf.height; ++y) {
        for (u32 x = 0; x < buf.width; ++x) {
            const u8* p = &linear[static_cast<size_t>(y) * buf.pitch + x * bpp];
            u8* o = &rgb[(static_cast<size_t>(y) * buf.width + x) * 3];
            if (bpp == 2) {
                const u16 v = static_cast<u16>(p[0] | (p[1] << 8));
                o[0] = static_cast<u8>(((v >> 11) & 31) * 255 / 31);
                o[1] = static_cast<u8>(((v >> 5) & 63) * 255 / 63);
                o[2] = static_cast<u8>((v & 31) * 255 / 31);
            } else if (buf.format == 5) { // BGRA
                o[0] = p[2]; o[1] = p[1]; o[2] = p[0];
            } else {
                o[0] = p[0]; o[1] = p[1]; o[2] = p[2];
            }
        }
    }
    return WritePng(path, buf.width, buf.height, rgb);
}

} // namespace

void CaptureAsync(std::function<void()> job) {
    Writer& w = W();
    {
        std::lock_guard lock(w.mu);
        if (w.jobs.size() >= kMaxQueued) {
            Log("capture: %zu captures still being written, this one is dropped", w.jobs.size());
            return;
        }
        w.jobs.push_back(std::move(job));
        if (!w.started) {
            w.started = true;
            for (u32 i = Writers(); i; --i) std::thread(WriterMain).detach();
        }
    }
    w.work.notify_one();
}

void FinishCaptures() {
    Writer& w = W();
    std::unique_lock lock(w.mu);
    if (!w.idle.wait_for(lock, std::chrono::seconds(30), [&] { return w.jobs.empty() && !w.busy; })) {
        Log("capture: %zu captures still unwritten after 30 s", w.jobs.size() + w.busy);
    }
}

void CapturePng(const PresentedBuffer& buf, const std::string& path) {
    const u32 gobs_per_block = 1u << buf.block_height_log2, block_px = 8u << buf.block_height_log2;
    const u64 size = u64{buf.pitch / 64} * ((buf.height + block_px - 1) / block_px) * gobs_per_block * 512;
    std::vector<u8> tiled(size);
    g_mem.Read(buf.address, tiled.data(), size); // now: the game may draw into the buffer once it is released
    CaptureAsync([buf, path, tiled = std::move(tiled)] {
        if (!DeswizzleToPng(buf, tiled, path)) Log("capture: could not write %s", path.c_str());
    });
}

bool WriteBgraPng(const std::string& path, u32 width, u32 height, const u8* bgra) {
    std::vector<u8> rgb(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        rgb[i * 3 + 0] = bgra[i * 4 + 2];
        rgb[i * 3 + 1] = bgra[i * 4 + 1];
        rgb[i * 3 + 2] = bgra[i * 4 + 0];
    }
    return WritePng(path, width, height, rgb);
}

} // namespace rt
