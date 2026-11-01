// Tap mode research captures:
// what the real NVN driver gets and sends, saved for offline study. They read
// NVN's internal formats (SDK 12.3.2, BRO), so they live in tap mode only and
// never feed the layer.
//
// - Shaders: at each nvnProgramSetShaders, every stage's control section
//   (0x800 bytes) and its whole code memory (up to control +0x704: tag, SPH,
//   instructions, compiler constants; from the driver's
//   GPU address through the pools the memory shadows saw) are written once per
//   control-section content to nvn_shaders/<hash>.ctrl / .code; every call
//   adds a line per stage to nvn_shaders/index.tsv (hash, stage index, stage
//   count, program, code GPU address, its guest address, control bytes, frame).
// - Command lists: RT_NVN_CAPTURE=<first frame>:<frames> saves every list
//   submitted with nvnQueueSubmitCommands during those frames to
//   nvn_submits.bin (format below), for replaying offline.
//
// NVN's control memory (read from the SDK, 2026-10-15): a handle is either
// packed — (address of the first segment entry) | (entry count << 48) | 1 —
// or the list header + 16, followed by entries: kind 3 = group {u8 3, u32
// count at +4}, then count segments of 24 bytes {u64 GPU address, u64 CPU
// address, u32 words, u8 flags}; kind 4 = jump {u64 next at +8}; kind 2 =
// end; kinds 5-22 = control commands of fixed sizes (5 FenceSync, 6 WaitSync).
#include "nvn.h"

#include "../perf.h"
#include "../threads.h"
#include "tap.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <filesystem>
#include <set>
#include <vector>

namespace rt::nvn {

namespace {

// --- Shaders ----------------------------------------------------------------------------------------

constexpr u64 kControlBytes = 0x800;
constexpr u64 kCodeLimit = 0x100000; // a sanity bound on control +0x704

std::mutex g_shaders_mu;
std::set<u64> g_shaders_seen;

u64 Fnv1a(const std::vector<u8>& bytes) {
    u64 h = 0xcbf29ce484222325ULL;
    for (u8 b : bytes) h = (h ^ b) * 0x100000001b3ULL;
    return h;
}

/// Reads up to `size` bytes at `va`, stopping at the end of its memory region.
std::vector<u8> ReadClamped(u64 va, u64 size) {
    const Region* r = g_mem.Find(va);
    if (!r) return {};
    std::vector<u8> v(std::min(size, r->base + r->size - va));
    g_mem.Read(va, v.data(), v.size());
    return v;
}

void Save(const std::string& path, const std::vector<u8>& bytes) {
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    }
}

void ShadowSetShaders(const std::string&, const GuestContext& before, const GuestContext&) {
    const u64 program = before.x[0], data = before.x[2];
    const u32 count = static_cast<u32>(before.x[1]);
    std::lock_guard lock(g_shaders_mu);
    for (u32 i = 0; i < count && i < 6; ++i) {
        const u64 code_gpu = hle::ReadGuest<u64>(data + 16ULL * i), control = hle::ReadGuest<u64>(data + 16ULL * i + 8);
        const std::vector<u8> ctrl = ReadClamped(control, kControlBytes);
        const u64 hash = Fnv1a(ctrl);
        const u64 code = DriverGpuToGuest(code_gpu);
        std::error_code ec;
        std::filesystem::create_directories("nvn_shaders", ec);
        if (g_shaders_seen.insert(hash).second) {
            // the stage's whole code memory: debug tag, SPH, instructions, compiler constants (control +0x704)
            const u64 used = ctrl.size() >= 0x708 ? std::min<u64>(hle::ReadGuest<u32>(control + 0x704), kCodeLimit) : 0;
            const std::vector<u8> code_bytes = code ? ReadClamped(code, used) : std::vector<u8>{};
            char name[64];
            std::snprintf(name, sizeof name, "nvn_shaders/%016llx", (unsigned long long)hash);
            Save(std::string(name) + ".ctrl", ctrl);
            Save(std::string(name) + ".code", code_bytes);
        }
        if (FILE* f = std::fopen("nvn_shaders/index.tsv", "a")) { // every call, repeats included
            std::fprintf(f, "%016llx\t%u\t%u\t0x%llx\t0x%llx\t0x%llx\t%zu\t%llu\n", (unsigned long long)hash, i,
                         count, (unsigned long long)program, (unsigned long long)code_gpu, (unsigned long long)code,
                         ctrl.size(), (unsigned long long)perf::Frames());
            std::fclose(f);
        }
    }
}

// --- Submitted command lists ---------------------------------------------------------------------------
// nvn_submits.bin: "NVSB" u32 version 1; per submission: "SUBM" u32 frame u32 lists; per list: "LIST"
// u32 entries; per entry: u32 kind; kind 3 (a segment): u64 GPU address, u64 CPU address, u32 words, then the
// words; any other kind (a control command): u32 size, then its bytes.

std::mutex g_submit_mu;
FILE* g_submits = nullptr;
u64 g_capture_first = 0, g_capture_frames = 0;

u32 ControlSize(u8 kind) {
    switch (kind) {
    case 5: case 6: case 7: case 8: case 9: case 10: case 11: case 14: case 15: case 18: case 19: case 20: return 16;
    case 12: case 13: return 24;
    case 21: return 40;
    case 22: return 8;
    default: return 0;
    }
}

void Put(const void* p, size_t n) { std::fwrite(p, 1, n, g_submits); }
void Put32(u32 v) { Put(&v, 4); }
void Put64(u64 v) { Put(&v, 8); }

void PutSegment(u64 entry) {
    const u64 gpu = hle::ReadGuest<u64>(entry), cpu = hle::ReadGuest<u64>(entry + 8);
    const u32 words = hle::ReadGuest<u32>(entry + 16);
    std::vector<u8> data(4ULL * words);
    if (!data.empty()) g_mem.Read(cpu, data.data(), data.size());
    Put32(3);
    Put64(gpu);
    Put64(cpu);
    Put32(words);
    Put(data.data(), data.size());
}

/// The entries of one list; returns how many were written.
u32 PutList(u64 handle) {
    u32 n = 0;
    if (handle & 1) { // packed: one group
        const u64 first = handle & 0x0000FFFFFFFFFFFEULL;
        const u32 count = static_cast<u32>(handle >> 48);
        for (u32 i = 0; i < count; ++i, ++n) PutSegment(first + 24ULL * i);
        return n;
    }
    for (u64 at = handle; n < 100000;) {
        const u8 kind = hle::ReadGuest<u8>(at);
        if (kind == 2) return n;
        if (kind == 4) {
            at = hle::ReadGuest<u64>(at + 8);
            continue;
        }
        if (kind == 3) {
            const u32 count = hle::ReadGuest<u32>(at + 4);
            at += 8;
            for (u32 i = 0; i < count; ++i, at += 24, ++n) PutSegment(at);
            continue;
        }
        const u32 size = ControlSize(kind);
        if (!size) { // saved raw for study (entry kind 0xffff: u32 size, from the list header on)
            Log("nvn capture: list 0x%llx: unknown control kind %u at 0x%llx; list cut there (raw bytes saved)",
                (unsigned long long)handle, kind, (unsigned long long)at);
            const std::vector<u8> raw = ReadClamped(handle - 0x10, 0x200);
            Put32(0xffff);
            Put32(static_cast<u32>(raw.size()));
            Put(raw.data(), raw.size());
            return n + 1;
        }
        std::vector<u8> bytes(size);
        g_mem.Read(at, bytes.data(), size);
        Put32(kind);
        Put32(size);
        Put(bytes.data(), size);
        at += size;
        ++n;
    }
    return n;
}

void ShadowSubmit(const std::string&, const GuestContext& before, const GuestContext&) {
    const u64 frame = perf::Frames();
    if (frame < g_capture_first || frame >= g_capture_first + g_capture_frames) return;
    const u32 count = static_cast<u32>(before.x[1]);
    std::lock_guard lock(g_submit_mu);
    if (!g_submits) {
        g_submits = std::fopen("nvn_submits.bin", "wb");
        if (!g_submits) return;
        Put("NVSB", 4);
        Put32(1);
        Log("nvn capture: saving the command lists of frames %llu..%llu to nvn_submits.bin",
            (unsigned long long)g_capture_first, (unsigned long long)(g_capture_first + g_capture_frames - 1));
    }
    Put("SUBM", 4);
    Put32(static_cast<u32>(frame));
    Put32(count);
    for (u32 i = 0; i < count; ++i) {
        const u64 handle = hle::ReadGuest<u64>(before.x[2] + 8ULL * i);
        Put("LIST", 4);
        const long at = std::ftell(g_submits);
        Put32(0);
        const u32 n = PutList(handle);
        const long end = std::ftell(g_submits);
        std::fseek(g_submits, at, SEEK_SET);
        Put32(n);
        std::fseek(g_submits, end, SEEK_SET);
    }
    std::fflush(g_submits);
}

} // namespace

void InstallCaptureShadows() {
    tap::AddShadow("nvnProgramSetShaders", ShadowSetShaders);
    if (const char* e = std::getenv("RT_NVN_CAPTURE")) {
        unsigned long long first = 0, frames = 0;
        if (std::sscanf(e, "%llu:%llu", &first, &frames) != 2) Fatal("RT_NVN_CAPTURE=%s: expected <first frame>:<frames>", e);
        g_capture_first = first, g_capture_frames = frames;
        tap::AddShadow("nvnQueueSubmitCommands", ShadowSubmit);
    }
}

} // namespace rt::nvn
