// Draws.
//
// Recording (signatures checked in BRO, 2026-10-16): DrawArrays(mode, first,
// count), DrawArraysInstanced(mode, first, count, base instance, instances),
// DrawElements(mode, index type, count, index address),
// DrawElementsBaseVertex(..., base vertex), DrawElementsInstanced(..., base
// vertex, base instance, instances): one record each.
//
// Translation: the bound render targets as one framebuffer (render.cpp), the
// pipeline for the bound state (pipeline.cpp), its descriptors pushed in
// drippu's order: constant buffers (c0 and c2 from the queue's shadows, c1 =
// the stage's compiler constants from its code memory, c3 + n = the game's
// uniform buffers), storage buffers from c0's tables, textures from the
// handles in c2 through the texture and sampler pools. Guest memory (vertex,
// index, uniform and storage data) is copied as the GPU sees it at that point
// of the submission: as at submit (5.0 rule 1) with the earlier GPU writes laid
// over (rule 4): those the CPU knows (fills, inline data, events) by the CPU,
// those only the GPU has (download-ring data: texture-to-buffer copies, shader
// stores) by a GPU copy from their ring slice, recorded before the draw.
// Writes of earlier submissions still in flight (not in guest memory yet) are
// laid over first, the same way; their download rings stay until the reading
// submission retires (queue.cpp). Viewport 0 and scissor 0 (NVN's upper-left window origin: a flipped
// viewport, as SSBU-recomp does).
//
// Dispatches: the compute pipeline for the bound compute program
// (pipeline.cpp), its descriptors as a stage's (compute's c0 built per
// dispatch from its bindings and group counts, report §8.1; its own c2), then
// vkCmdDispatch. Storage buffers a shader writes live in the download ring
// and reach guest memory at retirement; later reads of them in the same
// submission copy them on the GPU (above). Storage images through the texture
// pool.
//
// What the path does not handle yet throws NotYet: the run stops with it, or
// with RT_NVN_SKIP_DRAWS=1 the draw or dispatch is skipped and the reason
// logged once.
// drippu's headers first: vk.h brings X11's `Success` macro, which breaks its NvResult::Success.
#include "video_core/renderer_vulkan/vk_update_descriptor.h"

#include "../perf.h"
#include "../writetrack.h"
#include "heavy.h"
#include "pipeline.h"
#include "shaders.h"
#include "state.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <chrono>
#include <set>
#include <sstream>
#include <tuple>

#include "shader_recompiler/backend/spirv/emit_spirv.h"

namespace rt::nvn {

namespace {

struct DrawRecord {
    u32 mode;
    u32 indexed;    // 0: arrays; else the index type + 1
    s32 first;      // arrays: the first vertex; elements: the base vertex
    s32 count;
    u64 index_address;
    s32 base_instance, instances;
};

void RecordDraw(GuestContext* c, u64 cmdbuf, const DrawRecord& r) { Record(c, cmdbuf, Op::kDraw, r); }

struct IndirectDrawRecord { // Op::kDrawIndirect
    u32 mode;
    u32 indexed; // as DrawRecord
    u64 index_address;
    u64 indirect; // the parameters' address
};

} // namespace

NVN_PROC(nvnCommandBufferDrawArrays, void, u64 cmdbuf, u32 mode, s32 first, s32 count) {
    RecordDraw(c, cmdbuf, {mode, 0, first, count, 0, 0, 1});
}
NVN_PROC(nvnCommandBufferDrawArraysInstanced, void, u64 cmdbuf, u32 mode, s32 first, s32 count, s32 base_instance,
         s32 instances) {
    RecordDraw(c, cmdbuf, {mode, 0, first, count, 0, base_instance, instances});
}
NVN_PROC(nvnCommandBufferDrawElements, void, u64 cmdbuf, u32 mode, u32 type, s32 count, u64 address) {
    RecordDraw(c, cmdbuf, {mode, type + 1, 0, count, address, 0, 1});
}
NVN_PROC(nvnCommandBufferDrawElementsBaseVertex, void, u64 cmdbuf, u32 mode, u32 type, s32 count, u64 address,
         s32 base_vertex) {
    RecordDraw(c, cmdbuf, {mode, type + 1, base_vertex, count, address, 0, 1});
}
NVN_PROC(nvnCommandBufferDrawElementsInstanced, void, u64 cmdbuf, u32 mode, u32 type, s32 count, u64 address,
         s32 base_vertex, s32 base_instance, s32 instances) {
    RecordDraw(c, cmdbuf, {mode, type + 1, base_vertex, count, address, base_instance, instances});
}

// DrawArraysIndirect(mode, parameters) and DrawElementsIndirect(mode, index type, index address,
// parameters) (BRO 0x5849c0, 0x584be0: the driver hands the address to a GPU macro). The parameters, as
// Vulkan's (drippu's macro HLE reads them so): arrays {count, instances, first vertex, base instance},
// elements {count, instances, first index, base vertex, base instance}. Read as the GPU sees them at
// translation; parameters (in part) only the GPU has — a shader of an unretired submission wrote them — are
// copied on the GPU and drawn with vkCmdDraw(Indexed)Indirect (TranslateDrawRecord, Draw's gpu_params).
NVN_PROC(nvnCommandBufferDrawArraysIndirect, void, u64 cmdbuf, u32 mode, u64 indirect) {
    Record(c, cmdbuf, Op::kDrawIndirect, IndirectDrawRecord{mode, 0, 0, indirect});
}
NVN_PROC(nvnCommandBufferDrawElementsIndirect, void, u64 cmdbuf, u32 mode, u32 type, u64 address, u64 indirect) {
    Record(c, cmdbuf, Op::kDrawIndirect, IndirectDrawRecord{mode, type + 1, address, indirect});
}

// DispatchCompute(x, y, z): a control command in NVN (kind 14, 16 bytes), so 8 bytes of payload here: x in
// 32 bits, y and z in 16 (the Maxwell launch descriptor's widths; larger counts stop the run).
NVN_PROC(nvnCommandBufferDispatchCompute, void, u64 cmdbuf, s32 x, s32 y, s32 z) {
    if (x < 0 || y < 0 || y > 0xFFFF || z < 0 || z > 0xFFFF) Fatal("nvn: DispatchCompute(%d, %d, %d)", x, y, z);
    const u64 packed = static_cast<u64>(static_cast<u32>(x)) | static_cast<u64>(y) << 32 | static_cast<u64>(z) << 48;
    ControlRecord(c, cmdbuf, Op::kDispatch, packed, 16);
}

// DispatchComputeIndirect(parameters): the group counts {x, y, z} at `indirect`, read as the GPU sees them at
// translation; counts (in part) only the GPU has are copied on the GPU, into the dispatch's c0 too, and
// dispatched with vkCmdDispatchIndirect (as the indirect draws). Brothership has one call site (BRO
// 0x3168d8c, UE4's RHIDispatchIndirectComputeShader inferred), not reached in play so far (2026-10-23).
NVN_PROC(nvnCommandBufferDispatchComputeIndirect, void, u64 cmdbuf, u64 indirect) {
    Record(c, cmdbuf, Op::kDispatchIndirect, indirect);
}

namespace {

/// A part of a read that only the GPU has (download-ring data), or CPU-known data that must land after
/// such a part (uploaded): copied on the GPU, in order.
struct GpuPart {
    VkBuffer src;
    VkBufferCopy copy;   // dstOffset relative to the read's start
    bool upload = false; // Overlay: CPU-known bytes uploaded over an earlier GPU part (not a GPU write)
};

std::atomic<u64> g_gpu_reads{0}, g_earlier_reads{0};

// RT_PROFILE_DRAWS=1: where a draw's translation time goes. Each phase's time, the guest bytes it copies
// (snapshots that missed the per-submission cache, constant buffers) and the earlier writes it scans; logged
// as averages per draw every 50k draws, then reset.
enum Phase : u32 { kTargets, kPipeline, kCbuf, kStorage, kTexel, kTextures, kImages, kVertex, kIndex, kRecord, kBook, kPhases };
constexpr const char* kPhaseNames[kPhases] = {"targets", "pipeline", "cbufs", "storage", "texel", "textures",
                                              "images", "vertex", "index", "record", "bookkeeping"};
// Within each phase, the parts that read guest memory (which part a cache removes):
// the guest read, laying earlier writes over it (Overlay, KeepVisibleParts), the F9 note, the GPU copies of GPU parts;
// the phase's rest is reported as "rest" (for texel: the view and descriptors).
// "lookup" and "view": a buffer texture's pool entry and format checks, its texel view.
// "page search": PageCopy looking for a block that holds the read; "write setup": ShaderWrite's slices, write-back
// entry and device copy (its fill is timed as the parts above).
enum Part : u32 { kGuestRead, kOverlay, kVisible, kNote, kGpuCopies, kRangeStats, kLookup, kView, kPageSearch, kWriteSetup, kParts };
constexpr const char* kPartNames[kParts] = {"guest read", "overlay", "visible parts", "note", "GPU copies", "range stats",
                                            "lookup", "view", "page search", "write setup"};
struct DrawProfile {
    std::atomic<u64> ns[kPhases]{}, bytes[kPhases]{}, scanned{0}, draws{0}, snap_hits{0}, snap_misses{0};
    std::atomic<u64> part_ns[kPhases][kParts]{};
    std::atomic<u64> snap_stale{0}, snap_placed{0}; // misses: a write over the copy; a copy at an unusable offset
    std::atomic<u64> vertex_bound{0}, vertex_ranged{0}, vertex_streams{0}; // bound bytes, streams copied in part, all
    std::atomic<u64> page_hits{0}, page_new{0}, page_merged{0}, page_bytes{0}, page_fallback{0}; // PageCopy
    std::atomic<u64> page_stale{0}, page_outside{0}, page_placed{0}; // PageCopy: why a block was not used
    std::atomic<u64> page_stale_by[7]{};                              // stale: by the WriteBack::Kind written
    std::atomic<u64> page_unaligned[kPhases]{}, page_unaligned_bytes[kPhases]{}; // fell back: the address's alignment
    // CachedRead (Phase 2): reads served, pages reused / copied, bytes copied, why a read was not served
    std::atomic<u64> cache_reads{0}, cache_pages_reused{0}, cache_pages_copied{0}, cache_bytes{0}, cache_written{0},
        cache_full{0}, cache_placed{0}, cache_entries{0}, cache_evicted{0}, cache_checked{0}, cache_differ{0},
        cache_missed{0};
};
DrawProfile g_profile;
thread_local Phase t_phase = kRecord; // the phase the copies and scans count to

bool ProfileDraws() {
    static const bool on = std::getenv("RT_PROFILE_DRAWS") != nullptr;
    return on;
}

/// The phase and part timers run for RT_PROFILE_DRAWS and for RT_PROFILE_HEAVY (heavy.h: t_heavy_phases).
bool PhaseTimersOn() {
    static const bool on = ProfileDraws() || HeavyOn();
    return on;
}
static_assert(kPhases == kHeavyPhases && kParts == kHeavyParts, "heavy.h's copies of the phase and part counts");

u64 ProfileNow() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct PhaseTimer { // times its scope as `phase` (nothing when profiling is off)
    Phase phase, outer;
    u64 t;
    explicit PhaseTimer(Phase p) : phase(p), outer(t_phase), t(PhaseTimersOn() ? ProfileNow() : 0) { t_phase = p; }
    ~PhaseTimer() {
        t_phase = outer;
        if (!t) return;
        const u64 ns = ProfileNow() - t;
        g_profile.ns[phase].fetch_add(ns, std::memory_order_relaxed);
        if (t_heavy_phases) (*t_heavy_phases)[phase][0] += ns;
    }
};

struct PartTimer { // times its scope as `part` of the current phase (nothing when profiling is off)
    Part part;
    u64 t;
    explicit PartTimer(Part p) : part(p), t(PhaseTimersOn() ? ProfileNow() : 0) {}
    ~PartTimer() {
        if (!t) return;
        const u64 ns = ProfileNow() - t;
        g_profile.part_ns[t_phase][part].fetch_add(ns, std::memory_order_relaxed);
        if (t_heavy_phases) (*t_heavy_phases)[t_phase][1 + part] += ns;
    }
};

void ProfileBytes(u64 n) {
    if (ProfileDraws()) g_profile.bytes[t_phase].fetch_add(n, std::memory_order_relaxed);
}

void ProfileDrawDone() {
    if (!ProfileDraws()) return;
    const u64 n = g_profile.draws.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n % 50000) return;
    std::string line, parts;
    double total = 0;
    for (u32 i = 0; i < kPhases; ++i) {
        const double us = g_profile.ns[i].exchange(0) / 1e3 / 50000, kb = g_profile.bytes[i].exchange(0) / 1024.0 / 50000;
        total += us;
        std::string phase_parts;
        double in_parts = 0;
        for (u32 k = 0; k < kParts; ++k) {
            const double p = g_profile.part_ns[i][k].exchange(0) / 1e3 / 50000;
            in_parts += p;
            if (p < 0.05) continue;
            char b[64];
            std::snprintf(b, sizeof b, "%s%s %.1f", phase_parts.empty() ? "" : ", ", kPartNames[k], p);
            phase_parts += b;
        }
        if (!phase_parts.empty()) {
            char b[48];
            std::snprintf(b, sizeof b, ", rest %.1f", us - in_parts);
            parts += std::string(parts.empty() ? "" : "; ") + kPhaseNames[i] + ": " + phase_parts + b;
        }
        char b[96];
        std::snprintf(b, sizeof b, "; %s %.1f us%s", kPhaseNames[i], us, kb >= 0.05 ? (" " + std::to_string(int(kb * 10 + 0.5) / 10.0).substr(0, 5) + " KB").c_str() : "");
        line += b;
    }
    const u64 hits = g_profile.snap_hits.exchange(0), misses = g_profile.snap_misses.exchange(0);
    const u64 stale = g_profile.snap_stale.exchange(0), placed = g_profile.snap_placed.exchange(0);
    {
        const u64 bound = g_profile.vertex_bound.exchange(0), ranged = g_profile.vertex_ranged.exchange(0),
                  all = g_profile.vertex_streams.exchange(0);
        char b[96];
        std::snprintf(b, sizeof b, " (of %.1f KB bound; %llu of %llu streams copied in part)", bound / 1024.0 / 50000,
                      (unsigned long long)ranged, (unsigned long long)all);
        const size_t at = line.find(" KB", line.find("; vertex "));
        if (at != std::string::npos) line.insert(at + 3, b);
    }
    Log("nvn draw profile (t=%.1f s, per draw, %llu draws): %.1f us%s; earlier writes scanned %.0f; snapshot cache %llu hits / %llu misses "
        "(%llu stale, %llu at an unusable offset, the rest new in the submission)",
        Uptime(), (unsigned long long)n, total, line.c_str(), g_profile.scanned.exchange(0) / 50000.0, (unsigned long long)hits,
        (unsigned long long)misses, (unsigned long long)stale, (unsigned long long)placed);
    if (!parts.empty()) Log("nvn draw profile parts (us per draw): %s", parts.c_str());
    const u64 ph = g_profile.page_hits.exchange(0), pn = g_profile.page_new.exchange(0), pm = g_profile.page_merged.exchange(0),
              pb = g_profile.page_bytes.exchange(0), pf = g_profile.page_fallback.exchange(0);
    if (const u64 cr = g_profile.cache_reads.exchange(0), cw = g_profile.cache_written.exchange(0),
        cf = g_profile.cache_full.exchange(0), cp = g_profile.cache_placed.exchange(0); cr + cw + cf + cp) {
        Log("nvn draw profile buffer cache: %llu reads served (pages %llu reused, %llu copied: %.1f KB per draw); not served: "
            "%llu written in this or an earlier submission in flight, %llu no room, %llu placed too early; %llu entries made, "
            "%llu evicted; check: %llu compared, %llu differ (%llu pages MISSED)", (unsigned long long)cr,
            (unsigned long long)g_profile.cache_pages_reused.exchange(0), (unsigned long long)g_profile.cache_pages_copied.exchange(0),
            g_profile.cache_bytes.exchange(0) / 1024.0 / 50000, (unsigned long long)cw, (unsigned long long)cf,
            (unsigned long long)cp, (unsigned long long)g_profile.cache_entries.exchange(0),
            (unsigned long long)g_profile.cache_evicted.exchange(0), (unsigned long long)g_profile.cache_checked.load(),
            (unsigned long long)g_profile.cache_differ.load(), (unsigned long long)g_profile.cache_missed.load());
    }
    if (ph + pn + pf) {
        Log("nvn draw profile page copies: %llu reads all from filled pages, %llu reads filling pages, %llu blocks reserved, "
            "%.1f KB copied per draw, %llu reads fell back to their own copy; blocks passed over: %llu not holding the read, "
            "%llu placed too early, %llu stale (by the write's kind ring/bytes/fill/event/event add/counter: %llu/%llu/%llu/%llu/%llu/%llu)",
            (unsigned long long)ph, (unsigned long long)pn, (unsigned long long)pm, pb / 1024.0 / 50000, (unsigned long long)pf,
            (unsigned long long)g_profile.page_outside.exchange(0), (unsigned long long)g_profile.page_placed.exchange(0),
            (unsigned long long)g_profile.page_stale.exchange(0), (unsigned long long)g_profile.page_stale_by[0].exchange(0),
            (unsigned long long)g_profile.page_stale_by[1].exchange(0), (unsigned long long)g_profile.page_stale_by[2].exchange(0),
            (unsigned long long)g_profile.page_stale_by[3].exchange(0), (unsigned long long)g_profile.page_stale_by[4].exchange(0),
            (unsigned long long)g_profile.page_stale_by[5].exchange(0));
        std::string un;
        for (u32 i = 0; i < kPhases; ++i) {
            const u64 n = g_profile.page_unaligned[i].exchange(0), by = g_profile.page_unaligned_bytes[i].exchange(0);
            if (!n) continue;
            char b[80];
            (void)by;
            std::snprintf(b, sizeof b, "%s%s %llu", un.empty() ? "" : ", ", kPhaseNames[i], (unsigned long long)n);
            un += b;
        }
        if (!un.empty()) Log("nvn draw profile page copies: blocks shifted off the page grid for a read's alignment: %s", un.c_str());
    }
}

/// The GPU command behind a kRing write, for skip reasons.
const char* WriterName(const WriteBack& w) {
    switch (w.origin) {
    case WriteBack::Origin::kStorageBuffer: return w.stage == 5 ? "a compute shader's storage buffer" : "a graphics shader's storage buffer";
    case WriteBack::Origin::kBufferTexture: return w.stage == 5 ? "a compute shader's buffer texture" : "a graphics shader's buffer texture";
    case WriteBack::Origin::kCopyBuffer: return "CopyBufferToBuffer";
    case WriteBack::Origin::kCopyTexture: return "CopyTextureToBuffer";
    case WriteBack::Origin::kUnknown: break;
    }
    return "an unknown command";
}

/// Set while PageCopy fills whole pages: a pending counter report there makes the fill back off (NotYet), so the read
/// is copied on its own and only reads that really cover a report reach Overlay's kCounter case below.
thread_local bool t_page_fill = false;

/// A draw or dispatch reading a counter report whose submission has not retired (its value is made on the CPU at
/// retirement: queue.cpp): the read gets the bytes from before the report - in practice the previous report at that
/// address, as the game's CPU last saw it. Brothership entering Bowser's island (a tester's log, 2026-11-06) stopped
/// here; UE4 reads its reports on the CPU, so such a read most likely only spans the report. Logged, to find out.
void NoteStaleCounterRead(const WriteBack& w, u64 address, u64 size, bool earlier) {
    static std::atomic<u64> n{0};
    const u64 k = n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (k <= 20 || k % 10000 == 0) {
        Log("nvn draw: read 0x%llx+0x%llx covers a counter report at 0x%llx not retired yet (%s): the bytes from before "
            "it are used (#%llu)",
            (unsigned long long)address, (unsigned long long)size, (unsigned long long)w.address,
            earlier ? "an earlier submission" : "this submission", (unsigned long long)k);
    }
}

/// Lays the writes of `list` over `out` (guest memory [address, address + size)), in command order. A GPU
/// write (kRing) becomes a GpuPart; a CPU-known write over a GpuPart is uploaded and becomes one too.
/// Returns whether a GpuPart came from `list`'s ring.
bool Overlay(Translator& t, const std::vector<WriteBack>& list, u64 address, u64 size, u8* out, std::vector<GpuPart>* gpu,
             bool earlier) {
    bool used = false;
    if (ProfileDraws()) g_profile.scanned.fetch_add(list.size(), std::memory_order_relaxed);
    for (const WriteBack& w : list) {
        const u64 lo = std::max(address, w.address), hi = std::min(address + size, w.address + w.size);
        if (lo >= hi) continue;
        if (w.kind == WriteBack::Kind::kRing) {
            if (!gpu) {
                throw NotYet(std::string("a CPU-side read of data the GPU wrote (") + WriterName(w) +
                             (earlier ? ", an earlier submission)" : ", this submission)"));
            }
            gpu->push_back({w.ring_buffer, {w.ring_offset + (lo - w.address), lo - address, hi - lo}});
            used = true;
            continue;
        }
        bool over_gpu = false; // a CPU-known write after a GPU write to the same bytes
        if (gpu) {
            for (const GpuPart& g : *gpu) {
                over_gpu |= lo < address + g.copy.dstOffset + g.copy.size && address + g.copy.dstOffset < hi;
            }
        }
        switch (w.kind) {
        case WriteBack::Kind::kBytes: std::memcpy(out + (lo - address), w.bytes.data() + (lo - w.address), hi - lo); break;
        case WriteBack::Kind::kFill: { // the 32-bit value repeated from w.address: bytes up to a word, then words
            u64 a = lo;
            for (; a < hi && ((a - w.address) & 3); ++a) out[a - address] = static_cast<u8>(w.value >> (8 * ((a - w.address) & 3)));
            const u32 word = static_cast<u32>(w.value);
            for (; a + 4 <= hi; a += 4) std::memcpy(out + (a - address), &word, 4);
            for (; a < hi; ++a) out[a - address] = static_cast<u8>(w.value >> (8 * ((a - w.address) & 3)));
            break;
        }
        case WriteBack::Kind::kEventWrite:
        case WriteBack::Kind::kEventAdd: {
            if (lo != w.address || hi - lo != 4) throw NotYet("a draw reading part of an event written earlier in its submission");
            if (over_gpu && w.kind == WriteBack::Kind::kEventAdd) throw NotYet("an event add to a word the GPU wrote");
            u32 v = w.value;
            if (w.kind == WriteBack::Kind::kEventAdd) {
                u32 old;
                std::memcpy(&old, out + (lo - address), 4);
                v += old;
            }
            std::memcpy(out + (lo - address), &v, 4);
            break;
        }
        case WriteBack::Kind::kCounter: // `out` keeps what was there before the report (NoteStaleCounterRead)
            if (t_page_fill) throw NotYet("a page holding a counter report not retired yet");
            NoteStaleCounterRead(w, address, size, earlier);
            break;
        case WriteBack::Kind::kRing: break;
        }
        if (over_gpu) { // the bytes this write leaves (independent of what was there), after the GPU part
            const auto up = t.upload.Allocate(hi - lo, 4);
            std::memcpy(up.data, out + (lo - address), hi - lo);
            gpu->push_back({up.buffer, {up.offset, lo - address, hi - lo}, true});
        }
    }
    return used;
}

/// Keeps only what each part of `gpu` (in command order) leaves visible: a part's bytes that a later part covers
/// are dropped, so every byte is copied once, from its last writer. Same result as copying them all in order; the
/// 2.1 s cutscene frames (2026-10-24) read buffers ~240 chained dispatches had written, and copied every version:
/// 13 GB of GPU copies a frame for ~180 MB of reads. RT_NVN_ALL_GPU_PARTS=1: every part, as before (A/B).
void KeepVisibleParts(std::vector<GpuPart>& gpu) {
    static const bool all = [] {
        const char* e = std::getenv("RT_NVN_ALL_GPU_PARTS");
        return e && *e && *e != '0';
    }();
    if (all || gpu.size() < 2) return;
    std::map<u64, u64> covered; // [lo, hi) of the read, disjoint, by lo: what later parts write
    std::vector<GpuPart> kept;
    for (size_t i = gpu.size(); i-- > 0;) {
        const GpuPart& g = gpu[i];
        const u64 lo = g.copy.dstOffset, hi = lo + g.copy.size;
        u64 at = lo;
        auto it = covered.upper_bound(lo);
        if (it != covered.begin() && std::prev(it)->second > lo) --it;
        for (; at < hi; ++it) { // the gaps between covered ranges inside [lo, hi)
            const u64 next = it == covered.end() ? hi : std::min(hi, it->first);
            if (next > at) kept.push_back({g.src, {g.copy.srcOffset + (at - lo), at, next - at}, g.upload});
            if (it == covered.end()) break;
            at = std::max(at, it->second);
        }
        // [lo, hi) is covered from now on: merge it with the ranges it touches
        u64 mlo = lo, mhi = hi;
        auto m = covered.upper_bound(lo);
        if (m != covered.begin() && std::prev(m)->second >= lo) --m;
        while (m != covered.end() && m->first <= hi) {
            mlo = std::min(mlo, m->first), mhi = std::max(mhi, m->second);
            m = covered.erase(m);
        }
        covered[mlo] = mhi;
    }
    std::reverse(kept.begin(), kept.end()); // command order again (the pieces are disjoint now)
    gpu = std::move(kept);
}

/// Guest memory as the GPU sees it at this point of the submission (see the top), into CPU memory: the
/// writes of the earlier submissions still in flight, then this one's, in order. Parts only the GPU has go
/// to `gpu` (none allowed without it).
void ReadAsGpuSees(Translator& t, u64 address, u64 size, u8* out, std::vector<GpuPart>* gpu = nullptr) {
    ProfileBytes(size);
    {
        PartTimer timer(kGuestRead);
        g_mem.Read(address, out, size);
    }
    {
        PartTimer timer(kOverlay);
        for (Translator::Earlier& e : t.earlier) {
            if (Overlay(t, *e.writes, address, size, out, gpu, true)) {
                e.used = true;
                g_earlier_reads.fetch_add(1, std::memory_order_relaxed);
            }
        }
        Overlay(t, *t.writebacks, address, size, out, gpu, false);
    }
    PartTimer timer(kVisible);
    if (gpu) KeepVisibleParts(*gpu);
}

/// F9 draw log (LogDraw): the guest reads of the draw being translated, as " address+size#hash", the hash of
/// what the CPU knows of them ("+gpu<n>": n parts only the GPU has, not in the hash; "=": this submission's
/// earlier copy reused).
thread_local std::string* t_reads = nullptr;
/// RT_NVN_DUMP_F9=1, during an F9-logged dispatch: the file name prefix for its reads' data (NoteRead) and
/// its writes' (at retirement), f9/f<frame>_d<dispatch>_; empty otherwise.
thread_local std::string t_dump_prefix;
thread_local u32 t_dump_reads = 0;

void NoteRead(u64 address, u64 size, const u8* data, const std::vector<GpuPart>* gpu, bool reused) {
    if (!t_reads) return;
    PartTimer timer(kNote);
    if (!t_dump_prefix.empty() && data) {
        char name[64];
        std::snprintf(name, sizeof name, "r%u_%llx%s.bin", t_dump_reads++, (unsigned long long)address,
                      gpu && !gpu->empty() ? "_gpu" : "");
        if (FILE* f = std::fopen((t_dump_prefix + name).c_str(), "wb")) {
            std::fwrite(data, 1, size, f);
            std::fclose(f);
        }
    }
    const size_t gpu_parts = gpu ? gpu->size() : 0;
    u64 h = 0xcbf29ce484222325ull;
    for (u64 i = 0; i < size; ++i) h = (h ^ data[i]) * 0x100000001b3ull;
    char b[96];
    std::snprintf(b, sizeof b, " %s0x%llx+0x%llx#%08x", reused ? "=" : "", (unsigned long long)address,
                  (unsigned long long)size, static_cast<u32>(h ^ (h >> 32)));
    *t_reads += b;
    if (gpu_parts) {
        *t_reads += "+gpu" + std::to_string(gpu_parts) + "[";
        for (const GpuPart& g : *gpu) { // where in the read, and the size: what the GPU has
            std::snprintf(b, sizeof b, "%s0x%llx:0x%llx", &g == &gpu->front() ? "" : ",",
                          (unsigned long long)g.copy.dstOffset, (unsigned long long)g.copy.size);
            *t_reads += b;
        }
        *t_reads += "]";
    }
}

// --- The write index ---------------------------------------------------------------
//
// ReadAsGpuSees copies the whole read from guest memory, then lays every write of every submission in flight over
// it (scanning each list), then drops the GPU parts later ones cover (KeepVisibleParts). In the cutscenes' chains of
// ~975 dispatches that grows with the square of the dispatches, and 46-91 % of the copied bytes are then covered by
// GPU parts anyway (2026-10-28). IndexedRead gives each byte its latest writer from a per-submission index
// (WriteIndex, images.h) instead: guest memory only where nothing wrote, CPU-known writes laid, GPU parts directly.
// Same result as ReadAsGpuSees + KeepVisibleParts, except that bytes under a GPU part are not filled on the CPU and
// CPU-known bytes after a GPU write are in the slice, not uploaded as a part (the final bytes are the same).
// RT_NVN_WRITE_INDEX=0: off (A/B). RT_NVN_WRITE_INDEX_CHECK=1: both, compared (the old path's result is used).

bool WriteIndexOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_NVN_WRITE_INDEX");
        const bool v = !(e && *e == '0');
        Log("nvn: reads %s (RT_NVN_WRITE_INDEX)", v ? "through the write index" : "scan the write lists");
        return v;
    }();
    return on;
}

/// Sets [lo, hi) of a disjoint range map (lo -> {hi, value}) to `v`, trimming or splitting what it overlaps.
template <typename V>
void AssignRange(std::map<u64, std::pair<u64, V>>& m, u64 lo, u64 hi, const V& v) {
    if (lo >= hi) return;
    auto it = m.lower_bound(lo);
    if (it != m.begin()) {
        auto p = std::prev(it);
        if (p->second.first > lo) { // starts before lo, reaches into it: keep its head, and its tail past hi
            const auto [phi, pv] = p->second;
            p->second.first = lo;
            if (phi > hi) m.emplace(hi, std::pair<u64, V>{phi, pv});
        }
    }
    while (it != m.end() && it->first < hi) {
        if (it->second.first > hi) { // its tail survives
            const auto tail = it->second;
            it = m.erase(it);
            m.emplace(hi, tail);
            break;
        }
        it = m.erase(it);
    }
    m.emplace(lo, std::pair<u64, V>{hi, v});
}

/// Calls f(lo, hi, value) for the parts of the map's ranges inside [lo, hi), in address order.
template <typename V, typename F>
void ForRanges(const std::map<u64, std::pair<u64, V>>& m, u64 lo, u64 hi, F&& f) {
    auto it = m.upper_bound(lo);
    if (it != m.begin() && std::prev(it)->second.first > lo) --it;
    for (; it != m.end() && it->first < hi; ++it) {
        const u64 a = std::max(lo, it->first), b = std::min(hi, it->second.first);
        if (a < b) f(a, b, it->second.second);
    }
}

void CatchUp(WriteIndex& x, const std::vector<WriteBack>& list) {
    for (; x.indexed < list.size(); ++x.indexed) {
        const WriteBack& w = list[x.indexed];
        AssignRange<u32>(x.ranges, w.address, w.address + w.size, static_cast<u32>(x.indexed));
    }
}

/// ReadAsGpuSees + KeepVisibleParts through the write indices (see above), into `out` and `gpu` (empty on entry).
/// Returns false, having changed nothing, where a byte's latest write is an event or a counter report (their values
/// depend on what was there, or on retirement): the caller reads the old way.
bool IndexedRead(Translator& t, u64 address, u64 size, u8* out, std::vector<GpuPart>& gpu) {
    if (!t.index) return false;
    for (const Translator::Earlier& e : t.earlier) {
        if (!e.index) return false;
    }
    struct Writer {
        const WriteBack* w;
        size_t from; // t.earlier's index, t.earlier.size() for this submission
    };
    std::map<u64, std::pair<u64, Writer>> latest; // over [address, address + size), oldest list first
    const u64 end = address + size;
    {
        PartTimer timer(kOverlay);
        const auto lay = [&](WriteIndex& x, const std::vector<WriteBack>& list, size_t from) {
            CatchUp(x, list);
            ForRanges<u32>(x.ranges, address, end, [&](u64 a, u64 b, u32 k) { AssignRange(latest, a, b, Writer{&list[k], from}); });
        };
        for (size_t i = 0; i < t.earlier.size(); ++i) lay(*t.earlier[i].index, *t.earlier[i].writes, i);
        lay(*t.index, *t.writebacks, t.earlier.size());
        for (const auto& [lo, e] : latest) {
            const WriteBack::Kind k = e.second.w->kind;
            if (k == WriteBack::Kind::kEventWrite || k == WriteBack::Kind::kEventAdd || k == WriteBack::Kind::kCounter) return false;
        }
    }
    ProfileBytes(size);
    u64 at = address;
    std::vector<bool> marked(t.earlier.size(), false);
    for (const auto& [lo, e] : latest) {
        const u64 hi = e.first;
        const WriteBack& w = *e.second.w;
        if (lo > at) { // nothing wrote [at, lo): guest memory
            PartTimer timer(kGuestRead);
            g_mem.Read(at, out + (at - address), lo - at);
        }
        at = hi;
        switch (w.kind) {
        case WriteBack::Kind::kRing:
            gpu.push_back({w.ring_buffer, {w.ring_offset + (lo - w.address), lo - address, hi - lo}});
            if (e.second.from < t.earlier.size() && !marked[e.second.from]) {
                marked[e.second.from] = true;
                t.earlier[e.second.from].used = true;
                g_earlier_reads.fetch_add(1, std::memory_order_relaxed);
            }
            break;
        case WriteBack::Kind::kBytes: std::memcpy(out + (lo - address), w.bytes.data() + (lo - w.address), hi - lo); break;
        case WriteBack::Kind::kFill: { // as Overlay
            u64 a = lo;
            for (; a < hi && ((a - w.address) & 3); ++a) out[a - address] = static_cast<u8>(w.value >> (8 * ((a - w.address) & 3)));
            const u32 word = static_cast<u32>(w.value);
            for (; a + 4 <= hi; a += 4) std::memcpy(out + (a - address), &word, 4);
            for (; a < hi; ++a) out[a - address] = static_cast<u8>(w.value >> (8 * ((a - w.address) & 3)));
            break;
        }
        default: break; // (excluded above)
        }
    }
    if (at < end) {
        PartTimer timer(kGuestRead);
        g_mem.Read(at, out + (at - address), end - at);
    }
    return true;
}

/// RT_NVN_WRITE_INDEX_CHECK: the old path's result (`a`, `pa`) against IndexedRead's (`b`, `pb`) for one read. The
/// same GPU source for every byte a GPU part covers, the same CPU bytes everywhere else (an old upload part counts as
/// CPU bytes: they are `a`'s at its place). Returns the number of bytes that differ, and describes the first.
u64 CompareReads(const std::vector<u8>& a, const std::vector<GpuPart>& pa, const std::vector<u8>& b,
                 const std::vector<GpuPart>& pb, std::string& first) {
    struct Run {
        u64 lo, hi;
        VkBuffer src;
        u64 src_at_lo;
    };
    const auto runs = [](const std::vector<GpuPart>& p) {
        std::vector<Run> r;
        for (const GpuPart& g : p) {
            if (!g.upload) r.push_back({g.copy.dstOffset, g.copy.dstOffset + g.copy.size, g.src, g.copy.srcOffset});
        }
        std::sort(r.begin(), r.end(), [](const Run& x, const Run& y) { return x.lo < y.lo; });
        std::vector<Run> merged;
        for (const Run& x : r) {
            if (!merged.empty()) {
                Run& m = merged.back();
                if (m.hi == x.lo && m.src == x.src && m.src_at_lo + (m.hi - m.lo) == x.src_at_lo) {
                    m.hi = x.hi;
                    continue;
                }
            }
            merged.push_back(x);
        }
        return merged;
    };
    const std::vector<Run> ra = runs(pa), rb = runs(pb);
    u64 bad = 0;
    char line[200];
    const auto note = [&](const char* what, u64 off) {
        if (first.empty()) {
            std::snprintf(line, sizeof line, "%s at +0x%llx", what, (unsigned long long)off);
            first = line;
        }
    };
    if (ra.size() != rb.size()) {
        std::snprintf(line, sizeof line, "%zu GPU runs, the index %zu; ", ra.size(), rb.size());
        first = line;
    }
    for (size_t i = 0; i < std::min(ra.size(), rb.size()); ++i) {
        const Run &x = ra[i], &y = rb[i];
        if (x.lo != y.lo || x.hi != y.hi || x.src != y.src || x.src_at_lo != y.src_at_lo) {
            bad += std::max(x.hi, y.hi) - std::min(x.lo, y.lo);
            note("a GPU run differs", std::min(x.lo, y.lo));
        }
    }
    if (ra.size() != rb.size()) bad += 1;
    // CPU bytes outside the (old) GPU runs
    u64 at = 0;
    const auto cpu = [&](u64 lo, u64 hi) {
        for (u64 i = lo; i < hi; ++i) {
            if (a[i] != b[i]) {
                if (!bad) note("a CPU byte differs", i);
                ++bad;
            }
        }
    };
    for (const Run& x : ra) {
        cpu(at, x.lo);
        at = x.hi;
    }
    cpu(at, a.size());
    return bad;
}

/// ReadAsGpuSees into a ring slice, with the parts only the GPU has copied over it on the GPU (5.0 rule 4):
/// outside a render pass, then a barrier before what reads it. The writers ended with a full barrier (a
/// dispatch, a texture-to-buffer copy), which also orders earlier submissions on the queue before the copy.
/// Returns whether parts came from the GPU.
bool FillAsGpuSees(Translator& t, u64 address, u64 size, const UploadRing::Slice& slice) {
    static const bool all_parts = [] {
        const char* e = std::getenv("RT_NVN_ALL_GPU_PARTS");
        return e && *e && *e != '0';
    }();
    static const bool check = std::getenv("RT_NVN_WRITE_INDEX_CHECK") != nullptr;
    std::vector<GpuPart> gpu;
    // the write index (above), unless the F9 log hashes the whole read or every GPU part is wanted
    bool indexed = false;
    if (WriteIndexOn() && !t_reads && !all_parts && slice.data && size) {
        if (!check) {
            indexed = IndexedRead(t, address, size, slice.data, gpu);
        } else { // both, compared; the old path's result goes to the GPU
            thread_local std::vector<u8> a, b;
            a.resize(size), b.resize(size);
            std::vector<GpuPart> pb;
            ReadAsGpuSees(t, address, size, a.data(), &gpu);
            static std::atomic<u64> compared{0}, differ{0}, fell_back{0}, with_gpu{0}, gpu_bytes{0};
            if (!gpu.empty()) {
                with_gpu.fetch_add(1);
                for (const GpuPart& g : gpu) gpu_bytes.fetch_add(g.copy.size, std::memory_order_relaxed);
            }
            if (IndexedRead(t, address, size, b.data(), pb)) {
                std::string first;
                const u64 bad = CompareReads(a, gpu, b, pb, first);
                if (bad && differ.fetch_add(1) < 40) {
                    Log("nvn: WRITE INDEX CHECK: the read of 0x%llx+0x%llx (phase %u) differs from the scan: %llu bytes; %s; "
                        "old %zu GPU parts, index %zu", (unsigned long long)address, (unsigned long long)size,
                        static_cast<u32>(t_phase), (unsigned long long)bad, first.c_str(), gpu.size(), pb.size());
                }
            } else {
                fell_back.fetch_add(1);
            }
            if (const u64 n = compared.fetch_add(1) + 1; n % 20000 == 0) {
                Log("nvn: write index check: %llu reads compared (%llu with GPU parts, %.0f MB of them), %llu differ, %llu read the "
                    "old way (an event or report)", (unsigned long long)n, (unsigned long long)with_gpu.load(),
                    gpu_bytes.load() / 1048576.0, (unsigned long long)differ.load(), (unsigned long long)fell_back.load());
            }
            std::memcpy(slice.data, a.data(), size);
            indexed = true;
        }
    }
    if (!indexed) ReadAsGpuSees(t, address, size, slice.data, &gpu);
    NoteRead(address, size, slice.data, &gpu, false);
    if (gpu.empty()) return false;
    PartTimer timer(kGpuCopies);
    EndOpenPass(t.cmd);
    // In order: a part over bytes an earlier part copied waits for it (transfer to transfer). Copies without a
    // barrier between them may land in any order: the face morphs' buffer, written whole by two passes in a
    // frame (integer sums, then the float offsets), reached the draws with the sums on top in some frames
    // (2026-10-21: the results screen's flickering mouths and chins).
    for (size_t i = 0; i < gpu.size(); ++i) {
        GpuPart& g = gpu[i];
        for (size_t j = 0; j < i; ++j) {
            const VkBufferCopy& e = gpu[j].copy; // already moved into the slice
            const u64 lo = g.copy.dstOffset + slice.offset, hi = lo + g.copy.size;
            if (lo < e.dstOffset + e.size && e.dstOffset < hi) {
                t.cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                      VkMemoryBarrier{
                                          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                          .pNext = nullptr,
                                          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                          .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                      });
                break;
            }
        }
        g.copy.dstOffset += slice.offset;
        t.cmd.CopyBuffer(g.src, slice.buffer, g.copy);
        if (t.bytes) t.bytes->gpu_parts += g.copy.size;
    }
    FullBarrier(t.cmd);
    if (g_gpu_reads.fetch_add(1) < 4) {
        Log("nvn: a read of 0x%llx bytes at 0x%llx takes %zu part(s) the GPU wrote (copied on the GPU)",
            (unsigned long long)size, (unsigned long long)address, gpu.size());
    }
    return true;
}

/// Which bound textures and images are scaled: texture i of a stage is bit base + i of
/// words 0-3, image j bit base + j of words 4-5, each stage in its own range (shaders.h RescaleRange), so the
/// stages of a draw, which share the push constants, never share a bit.
struct RescaleBits {
    using Words = std::array<u32, Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS>;
    Words scaled{};
    void Note(const RescaleRange& range, bool image, u32 index, bool is_scaled) {
        if (!is_scaled) return;
        if (index >= (image ? range.images : range.textures)) { // the pass leaves it unscaled
            static std::atomic<bool> told{false};
            if (!told.exchange(true)) {
                Log("resolution: a scaled %s is bound past its stage's range: read as unscaled", image ? "image" : "texture");
            }
            return;
        }
        const u32 bit = (image ? 32 * Shader::Backend::SPIRV::NUM_TEXTURE_SCALING_WORDS + range.image_base
                               : range.texture_base) + index;
        scaled[bit / 32] |= 1u << (bit % 32);
    }
};

/// The rescaling range of NVN stage `nvn_stage` (program.h Stage: vertex, fragment, geometry, tessellation
/// control, evaluation, compute).
RescaleRange RescaleRangeOfNvn(u32 nvn_stage) {
    static constexpr Shader::Stage kStages[] = {Shader::Stage::VertexB, Shader::Stage::Fragment, Shader::Stage::Geometry,
                                                Shader::Stage::TessellationControl, Shader::Stage::TessellationEval,
                                                Shader::Stage::Compute};
    return RescaleRangeOf(kStages[std::min<u32>(nvn_stage, 5)]);
}

/// The push constants' bit masks: the scaled bindings, and in bits 124-127 of the texture mask the run's n (the
/// factor x 4, rescaling_pass.cpp).
RescaleBits::Words RescalingWords(const RescaleBits::Words& scaled) {
    RescaleBits::Words words = scaled;
    words[kRescaleFactorBit / 32] |= ScaleN() << (kRescaleFactorBit % 32);
    return words;
}

/// RT_NVN_RANGE_STATS=1 (a test aid): the copies Snapshot makes of guest memory, by
/// (address, size) with a hash of their bytes, sampled: frames 0-4 of every 60 are recorded, and the last of them is
/// compared with the three before it (the plaza's texel ranges were mostly at new addresses each frame, 2026-10-24:
/// a ring the game rotates?). Logged per sample ("nvn ranges: ..."): that frame's copies and bytes, its distinct
/// ranges, those copied 1, 2 or 3 frames before (the nearest) and whether byte-identical then (what a cache would not
/// copy again), and those with GPU-written parts (never cacheable, not hashed), by draw phase. The hash reads guest
/// memory again into a host buffer, never the upload slice: that is write-combined memory, and reading it back cost
/// 42 ms a frame for 3.8 MB (the first plaza test: 20 fps). Its time is RT_PROFILE_DRAWS' "range stats" part and
/// the line's "hashing". Also "the same bytes at any address": the plaza's texel ranges were new (address, size)
/// pairs each frame, so whether the content repeats elsewhere decides between a cache by address and none.
bool RangeStatsOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_NVN_RANGE_STATS");
        return e && *e && *e != '0';
    }();
    return on;
}

/// The finished sample's last frame against the three before it, logged.
struct RangeFrame {
    std::map<std::array<u64, 2>, std::array<u64, 3>> ranges; // (address, size): {hash, phase, gpu}
    std::map<u64, std::pair<u64, u32>> pages;               // 4 KB guest pages the non-GPU ranges cover: {hash, phase}
};

void RangeStatsLog(const std::vector<RangeFrame>& frames, u64 copies, u64 copy_bytes, u64 hash_ns) {
    std::vector<std::map<std::array<u64, 2>, std::array<u64, 3>>> history;
    for (const RangeFrame& f : frames) history.push_back(f.ranges);
    // pages: the last frame's against the latest earlier frame that covered each (the cache's view: page tracking)
    u64 pages = 0, pages_seen = 0, pages_same = 0, page_phase[kPhases]{}, page_phase_same[kPhases]{};
    for (const auto& [page, e] : frames.back().pages) {
        ++pages, ++page_phase[e.second];
        for (size_t f = frames.size() - 1; f-- > 0 && f + 4 >= frames.size();) {
            const auto it = frames[f].pages.find(page);
            if (it == frames[f].pages.end()) continue;
            ++pages_seen;
            if (it->second.first == e.first) ++pages_same, ++page_phase_same[e.second];
            break;
        }
    }
    std::string page_by;
    for (u32 p = 0; p < kPhases; ++p) {
        if (!page_phase[p]) continue;
        char b[80];
        std::snprintf(b, sizeof b, "%s%s %.1f MB (%.1f unchanged)", page_by.empty() ? "" : ", ", kPhaseNames[p],
                      page_phase[p] * 4096 / 1048576.0, page_phase_same[p] * 4096 / 1048576.0);
        page_by += b;
    }
    Log("nvn ranges: pages: %llu guest pages (%.1f MB) under the sampled frame's non-GPU copies; covered in the 3 frames "
        "before: %llu (%.1f MB); unchanged since then: %llu (%.1f MB); by phase: %s",
        (unsigned long long)pages, pages * 4096 / 1048576.0, (unsigned long long)pages_seen, pages_seen * 4096 / 1048576.0,
        (unsigned long long)pages_same, pages_same * 4096 / 1048576.0, page_by.c_str());
    const auto& last = history.back();
    u64 ranges = 0, bytes = 0, gpu = 0, gpu_bytes = 0, seen[3]{}, seen_bytes[3]{}, same[3]{}, same_bytes[3]{};
    u64 phase_bytes[kPhases]{}, phase_same[kPhases]{}, content = 0, content_bytes = 0, phase_content[kPhases]{};
    std::set<std::pair<u64, u64>> earlier; // (size, hash) of the three frames before: the same bytes at any address
    for (size_t f = history.size() >= 4 ? history.size() - 4 : 0; f + 1 < history.size(); ++f) {
        for (const auto& [key, r] : history[f]) if (!r[2]) earlier.insert({key[1], r[0]});
    }
    for (const auto& [key, r] : last) {
        if (!r[2] && earlier.count({key[1], r[0]})) ++content, content_bytes += key[1], phase_content[r[1]] += key[1];
        ++ranges, bytes += key[1], phase_bytes[r[1]] += key[1];
        if (r[2]) {
            ++gpu, gpu_bytes += key[1];
            continue;
        }
        for (u32 back = 1; back <= 3 && back < history.size(); ++back) { // the nearest frame that copied it
            const auto& then = history[history.size() - 1 - back];
            const auto it = then.find(key);
            if (it == then.end() || it->second[2]) continue;
            ++seen[back - 1], seen_bytes[back - 1] += key[1];
            if (it->second[0] == r[0]) ++same[back - 1], same_bytes[back - 1] += key[1], phase_same[r[1]] += key[1];
            break;
        }
    }
    const double mb = 1048576.0;
    std::string by;
    for (u32 p = 0; p < kPhases; ++p) {
        if (phase_bytes[p] / mb < 0.05) continue;
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %.1f MB (%.1f same, %.1f same bytes anywhere)", by.empty() ? "" : ", ", kPhaseNames[p],
                      phase_bytes[p] / mb, phase_same[p] / mb, phase_content[p] / mb);
        by += b;
    }
    Log("nvn ranges: t=%.1f s, a sampled frame: %llu copies %.1f MB; %llu distinct ranges %.1f MB; copied 1/2/3 frames before "
                "(nearest): %llu/%llu/%llu ranges, %.1f/%.1f/%.1f MB; of those byte-identical: %.1f/%.1f/%.1f MB; the same bytes "
        "at any address in those frames: %llu ranges %.1f MB; with GPU parts %llu (%.1f MB); by phase: %s; hashing %.2f ms "
        "a sampled frame",
        Uptime(), (unsigned long long)copies, copy_bytes / mb, (unsigned long long)ranges, bytes / mb,
        (unsigned long long)seen[0], (unsigned long long)seen[1], (unsigned long long)seen[2], seen_bytes[0] / mb,
        seen_bytes[1] / mb, seen_bytes[2] / mb, same_bytes[0] / mb, same_bytes[1] / mb, same_bytes[2] / mb,
        (unsigned long long)content, content_bytes / mb, (unsigned long long)gpu, gpu_bytes / mb, by.c_str(), hash_ns / 1e6 / history.size());
}

void RangeStatsNote(u64 address, u64 size, bool gpu) {
    constexpr u64 kCycle = 60, kSampled = 5;
    const u64 frame_now = perf::Frames();
    if (frame_now % kCycle >= kSampled) return;
    static std::mutex mu;
    static std::vector<RangeFrame> history; // the sample's frames, oldest first
    static u64 frame = ~0ull, hash_ns = 0, copies = 0, copy_bytes = 0;
    PartTimer timer(kRangeStats);
    const u64 t0 = ProfileNow();
    u64 h = 0x9e3779b97f4a7c15ull ^ size;
    thread_local std::vector<std::pair<u64, u64>> page_hashes; // this copy's whole pages: {page, hash}
    page_hashes.clear();
    if (!gpu && size) {
        thread_local std::vector<u8> copy;
        const u64 lo = address & ~u64{0xfff}, hi = (address + size + 0xfff) & ~u64{0xfff};
        copy.resize(hi - lo);
        g_mem.Read(lo, copy.data(), hi - lo); // the pages around it too (mapped: whole pages)
        for (u64 pg = lo; pg < hi; pg += 0x1000) {
            u64 ph = 0x9e3779b97f4a7c15ull;
            for (u64 k = 0; k < 0x1000; k += 8) {
                u64 w;
                std::memcpy(&w, copy.data() + (pg - lo) + k, 8);
                ph = (ph ^ w) * 0xff51afd7ed558ccdull, ph ^= ph >> 32;
            }
            page_hashes.push_back({pg, ph});
        }
        const u8* data = copy.data() + (address - lo);
        u64 i = 0;
        for (; i + 8 <= size; i += 8) {
            u64 w;
            std::memcpy(&w, data + i, 8);
            h = (h ^ w) * 0xff51afd7ed558ccdull, h ^= h >> 32;
        }
        for (; i < size; ++i) h = (h ^ data[i]) * 0x100000001b3ull;
    }
    const u64 took = ProfileNow() - t0;
    std::lock_guard lock(mu);
    if (frame_now != frame) {
        if (frame_now % kCycle == 0) { // a new sample: the last one, if complete, first
            if (history.size() == kSampled) RangeStatsLog(history, copies, copy_bytes, hash_ns);
            history.clear(), hash_ns = copies = copy_bytes = 0;
        }
        history.emplace_back();
        frame = frame_now;
    }
    hash_ns += took;
    if (frame_now % kCycle == kSampled - 1) ++copies, copy_bytes += size;
    auto& r = history.back().ranges[{address, size}];
    r[0] = h, r[1] = t_phase, r[2] = r[2] || gpu;
    for (const auto& [pg, ph] : page_hashes) history.back().pages[pg] = {ph, t_phase};
}

/// RT_NVN_PAGE_COPY (default on; 0: off, each read copied on its own as before - A/B).
bool PageCopyOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_NVN_PAGE_COPY");
        return !(e && *e == '0');
    }();
    return on;
}

/// The guest bytes [address, address + size) as an offset into this submission's page-aligned copy of their block
/// (Translator::spans): each page copied once per submission, when a read first needs it, instead of once per
/// (address, size) - the heavy scenes copied ~150 MB a frame from 13-21 MB of pages, through ~1,300 different
/// windows a frame over the same memory. Blocks are 256 KB-aligned runs of guest memory
/// (a read across a boundary gets a block of several); a block's room in the upload ring is reserved whole, its pages
/// filled on demand, and a page is never written again once filled (draws recorded before may read it): a read over
/// bytes written since their page was filled gets a new block. A block keeps every byte at its guest address mod
/// 4096 (aligned as on the console) unless the read that made it needs more alignment than its address has (`need`:
/// Vulkan's own alignment for the binding's offset): then the block is shifted by a few bytes, and reads it does not
/// suit get their own; `min_offset` is where the binding may start (Snapshot's). False: the read is copied on its own
/// (a page that cannot be copied: NotYet, e.g. a counter report the read itself never touches).
thread_local std::string t_page_note; // RT_NVN_PAGE_COPY_CHECK: how the last PageCopy served its read

bool PageCopy(Translator& t, u64 address, u64 size, u64 need, u64 min_offset, UploadRing::Slice& out) {
    constexpr u64 kPage = 0x1000, kBlock = 0x40000;
    static const bool noting = std::getenv("RT_NVN_PAGE_COPY_CHECK") != nullptr;
    if (!size) return false;
    const std::vector<WriteBack>& w = *t.writebacks;
    const u64 a = address & ~(kPage - 1), b = (address + size + kPage - 1) & ~(kPage - 1);
    const auto usable = [&](u64 lo, Translator::Span& s) { // holds the read, placed for it, nothing written over it since
        if (lo > address || s.hi < address + size) {
            if (ProfileDraws()) g_profile.page_outside.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (s.slice.offset + (address - lo) < min_offset || (s.slice.offset + (address - lo)) % need) {
            if (ProfileDraws()) g_profile.page_placed.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        for (; s.scanned < w.size(); ++s.scanned) {
            const u64 x = std::max(w[s.scanned].address, lo), y = std::min(w[s.scanned].address + w[s.scanned].size, s.hi);
            if (x < y) s.written.push_back({x, y, s.scanned});
        }
        for (const auto& r : s.written) {
            const u64 x = std::max(r[0], address), y = std::min(r[1], address + size);
            if (x >= y) continue;
            for (u64 pg = x & ~(kPage - 1); pg < y; pg += kPage) { // a page filled before that write: stale
                const size_t f = s.filled[(pg - lo) / kPage];
                if (f && f - 1 <= r[2]) {
                    if (ProfileDraws()) {
                        g_profile.page_stale.fetch_add(1, std::memory_order_relaxed);
                        g_profile.page_stale_by[static_cast<u32>(w[r[2]].kind) % 7].fetch_add(1, std::memory_order_relaxed);
                    }
                    return false;
                }
            }
        }
        return true;
    };
    Translator::Span* span = nullptr;
    u64 span_lo = 0;
    {
        PartTimer search(kPageSearch);
        for (auto it = t.spans.upper_bound(address); it != t.spans.begin();) {
            --it;
            if (it->first + t.longest_span <= address) break;
            if (usable(it->first, it->second)) {
                span = &it->second, span_lo = it->first;
                break;
            }
        }
    }
    bool created = false;
    if (!span) {
        // shifted off the page grid when the read's own address is not aligned as Vulkan needs (texel buffers of
        // ~1 MB at addresses not 16-aligned: 29 KB a draw on the plaza fell back before, 2026-10-24)
        const u64 lo = a & ~(kBlock - 1), hi = (b + kBlock - 1) & ~(kBlock - 1), skip = address - lo;
        const u64 shift = (need - skip % need) % need;
        if (shift && ProfileDraws()) g_profile.page_unaligned[t_phase].fetch_add(1, std::memory_order_relaxed);
        auto slice = t.upload.Allocate(hi - lo + shift, kPage, min_offset > skip + shift ? min_offset - skip - shift : 0);
        slice.offset += shift;
        if (slice.data) slice.data += shift;
        // its write list starts at the first write not done yet: a block made while a draw or dispatch is set up must
        // still see that command's own writes (pending_from) - made after them, it reused a compute pass's pre-pass
        // vertex data for the faces (2026-10-24, found with RT_NVN_PAGE_COPY_CHECK)
        const auto it = t.spans.insert(
            {lo, Translator::Span{hi, slice, std::vector<size_t>((hi - lo) / kPage, 0), std::min(w.size(), t.pending_from), {}}});
        t.longest_span = std::max(t.longest_span, hi - lo);
        span = &it->second, span_lo = lo, created = true;
    }
    if (noting) { // the block's state for this read before filling: pages' fill indices, the writes over the read
        char line[200];
        std::snprintf(line, sizeof line, "%s block 0x%llx, writes now %zu, pending from %lld; fill index per page:",
                      created ? "new" : "old", (unsigned long long)span_lo, w.size(),
                      t.pending_from == SIZE_MAX ? -1ll : (long long)t.pending_from);
        t_page_note = line;
        for (u64 pg = a; pg < b; pg += kPage) t_page_note += " " + std::to_string(span->filled[(pg - span_lo) / kPage]);
        t_page_note += "; writes over the read (index kind origin stage):";
        for (size_t i = 0; i < w.size(); ++i) {
            if (w[i].address < address + size && address < w[i].address + w[i].size) {
                std::snprintf(line, sizeof line, " %zu/%u/%u/%u", i, static_cast<u32>(w[i].kind), static_cast<u32>(w[i].origin),
                              static_cast<u32>(w[i].stage));
                t_page_note += line;
            }
        }
        t_page_note += "; earlier submissions' writes over it:";
        for (size_t k = 0; k < t.earlier.size(); ++k) {
            for (const WriteBack& x : *t.earlier[k].writes) {
                if (x.address < address + size && address < x.address + x.size) t_page_note += " s-" + std::to_string(t.earlier.size() - k);
            }
        }
    }
    // the read's pages not filled yet, in runs
    u64 filled_bytes = 0;
    for (u64 pg = a; pg < b;) {
        if (span->filled[(pg - span_lo) / kPage]) {
            pg += kPage;
            continue;
        }
        u64 end = pg;
        while (end < b && !span->filled[(end - span_lo) / kPage]) end += kPage;
        const UploadRing::Slice part{span->slice.buffer, span->slice.offset + (pg - span_lo),
                                     span->slice.data ? span->slice.data + (pg - span_lo) : nullptr};
        try {
            struct PageFill {
                PageFill() { t_page_fill = true; }
                ~PageFill() { t_page_fill = false; }
            } page_fill;
            FillAsGpuSees(t, pg, end - pg, part);
        } catch (const NotYet&) { // those pages stay unfilled (nothing read them yet): the read is copied on its own
            if (ProfileDraws()) g_profile.page_fallback.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        // (writes of the draw or dispatch being set up count as after the fill: their bytes change when it runs -
        // the faces' morph results were read as they were before their compute pass, 2026-10-24)
        for (u64 q = pg; q < end; q += kPage) span->filled[(q - span_lo) / kPage] = std::min(w.size(), t.pending_from) + 1;
        if (writetrack::CheckOn()) writetrack::Watch(pg, end - pg); // Phase 1's check under the game's writes
        filled_bytes += end - pg, pg = end;
    }
    if (t.bytes) t.bytes->read_fills += filled_bytes;
    if (ProfileDraws()) {
        (filled_bytes ? g_profile.page_new : g_profile.page_hits).fetch_add(1, std::memory_order_relaxed);
        if (created) g_profile.page_merged.fetch_add(1, std::memory_order_relaxed);
        g_profile.page_bytes.fetch_add(filled_bytes, std::memory_order_relaxed);
    }
    out = {span->slice.buffer, span->slice.offset + (address - span_lo),
           span->slice.data ? span->slice.data + (address - span_lo) : nullptr};
    return true;
}

// --- The buffer cache --------------------------------------------------------------

/// RT_NVN_BUFFER_CACHE (default on; 0: off - every submission copies its pages, Phase 0.5); RT_NVN_BUFFER_CACHE_MB:
/// the arena (default 512); RT_NVN_BUFFER_CACHE_CHECK=1 (test aid): a host shadow of every entry, and each served
/// read compared with guest memory as the GPU sees it.
bool CacheCheckOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_NVN_BUFFER_CACHE_CHECK");
        return e && *e && *e != '0';
    }();
    return on;
}

} // namespace

const char* HeavyPhaseName(u32 phase) { return phase < kPhases ? kPhaseNames[phase] : "?"; }
const char* HeavyPartName(u32 part) { return part < kParts ? kPartNames[part] : "?"; }

struct BufferCache {
    static constexpr u64 kGranule = 0x40000, kPage = 0x1000;
    struct Entry {
        u64 lo, hi;       // guest memory, 256 KB-aligned
        u64 offset;       // the arena offset of lo's byte (granule start + shift)
        u32 granule, granules;
        std::vector<u32> gen;     // per page: writetrack's generation when copied (Now() before its Watch)
        std::vector<u64> decided; // per page: the submission that last decided it (0: never copied)
        u64 last_used = 0;        // the last submission that read it (evicted only once that retired)
        u64 sub = 0;              // the submission `written` belongs to
        size_t scanned = 0;       // of that submission's writebacks
        std::vector<std::array<u64, 2>> written; // writes over [lo, hi) in flight before the copy's GPU execution
        std::vector<u8> shadow;   // RT_NVN_BUFFER_CACHE_CHECK
    };
    Vulkan::vk::Buffer arena;
    u64 granules = 0;
    std::vector<u8> used; // per granule
    std::multimap<u64, std::unique_ptr<Entry>> entries;
    u64 longest = 0;

    /// `n` free granules in a row (evicting entries no submission in flight uses, least recently used first); -1: none.
    s64 Allocate(u32 n, u64 retired) {
        for (int pass = 0; pass < 2; ++pass) {
            u32 run = 0;
            for (u64 g = 0; g < granules; ++g) {
                run = used[g] ? 0 : run + 1;
                if (run == n) {
                    for (u64 k = g + 1 - n; k <= g; ++k) used[k] = 1;
                    return static_cast<s64>(g + 1 - n);
                }
            }
            if (pass) break;
            std::vector<std::multimap<u64, std::unique_ptr<Entry>>::iterator> idle;
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                if (it->second->last_used <= retired) idle.push_back(it);
            }
            std::sort(idle.begin(), idle.end(), [](auto x, auto y) { return x->second->last_used < y->second->last_used; });
            // free the older half of the idle ones (at least enough granules), then look again
            u64 freed = 0;
            for (size_t i = 0; i < idle.size() && (freed < 2ull * n || i < idle.size() / 2); ++i) {
                const Entry& e = *idle[i]->second;
                for (u32 k = 0; k < e.granules; ++k) used[e.granule + k] = 0;
                freed += e.granules;
                entries.erase(idle[i]);
                if (ProfileDraws()) g_profile.cache_evicted.fetch_add(1, std::memory_order_relaxed);
            }
            if (!freed) break;
        }
        return -1;
    }
};

std::shared_ptr<BufferCache> MakeBufferCache() {
    const char* e = std::getenv("RT_NVN_BUFFER_CACHE");
    if (e && *e == '0') return nullptr;
    const char* mb = std::getenv("RT_NVN_BUFFER_CACHE_MB");
    const u64 bytes = (mb ? std::max<u64>(std::strtoull(mb, nullptr, 0), 16) : 512) << 20;
    auto c = std::make_shared<BufferCache>();
    c->granules = bytes / BufferCache::kGranule;
    c->used.assign(c->granules, 0);
    c->arena = Vk().allocator->CreateBuffer(
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = bytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        Vulkan::MemoryUsage::DeviceLocal);
    Log("nvn: buffer cache: %llu MB of device memory (RT_NVN_BUFFER_CACHE_MB; RT_NVN_BUFFER_CACHE=0: off)",
        (unsigned long long)(bytes >> 20));
    return c;
}

VkBuffer CacheBuffer(const BufferCache& cache) { return *cache.arena; }

namespace {

/// RT_NVN_BUFFER_CACHE_KINDS=<list of cbufs,storage,texel,vertex,index> (default all): which reads the cache serves.
/// RT_NVN_BUFFER_CACHE_BISECT=1 (test aid): the cache serves one kind at a time, cycling every 6 s through vertex,
/// index, texel, cbufs, storage and then all of them, each switch logged ("nvn: buffer cache bisect: ..."), to find
/// which kind of read a rendering error comes with from one play run's F7 flags (the faces, 2026-10-24).
bool CacheKindOn(Phase phase) {
    static const u32 mask = [] {
        const char* e = std::getenv("RT_NVN_BUFFER_CACHE_KINDS");
        if (!e || !*e) return ~0u;
        u32 m = 0;
        for (u32 p = 0; p < kPhases; ++p) {
            if (std::strstr(e, kPhaseNames[p])) m |= 1u << p;
        }
        Log("nvn: buffer cache serves only: %s (RT_NVN_BUFFER_CACHE_KINDS)", e);
        return m;
    }();
    static const bool bisect = [] {
        const char* e = std::getenv("RT_NVN_BUFFER_CACHE_BISECT");
        return e && *e && *e != '0';
    }();
    if (!bisect) return mask >> phase & 1;
    static const Phase order[] = {kVertex, kIndex, kTexel, kCbuf, kStorage};
    const u64 window = static_cast<u64>(Uptime() / 6);
    const u64 slot = window % 6; // 0-4: one kind, 5: all
    static std::atomic<u64> told{~0ull};
    if (u64 was = told.load(); was != window && told.compare_exchange_strong(was, window)) {
        Log("nvn: buffer cache bisect: t=%.1f s: now %s", Uptime(), slot == 5 ? "all kinds" : kPhaseNames[order[slot]]);
    }
    return slot == 5 || phase == order[slot];
}

/// The guest bytes [address, address + size) from the queue's buffer cache (BuferCache): pages unchanged since they
/// were copied (writetrack) are used as they are, the others copied again before this submission's commands (one
/// GPU copy of each such page, queue.cpp). False: not served; the caller
/// copies the read for this submission alone (PageCopy). `need`, `min_offset`: as PageCopy's.
bool CachedRead(Translator& t, u64 address, u64 size, u64 need, u64 min_offset, UploadRing::Slice& out) {
    using E = BufferCache::Entry;
    constexpr u64 G = BufferCache::kGranule, P = BufferCache::kPage;
    if (!t.cache || !size || t_reads) return false; // (the F9 draw log needs the bytes on the CPU)
    if (t_phase != kCbuf && t_phase != kStorage && t_phase != kTexel && t_phase != kVertex && t_phase != kIndex) return false;
    if (!CacheKindOn(t_phase)) return false;
    BufferCache& c = *t.cache;
    const std::vector<WriteBack>& w = *t.writebacks;
    E* e = nullptr;
    for (auto it = c.entries.upper_bound(address); it != c.entries.begin();) {
        --it;
        if (it->first + c.longest <= address) break;
        E& x = *it->second;
        const u64 at = x.offset + (address - x.lo);
        if (x.hi >= address + size && at % need == 0 && at >= min_offset) {
            e = &x;
            break;
        }
    }
    if (!e) {
        const u64 lo = address & ~(G - 1), hi = (address + size + G - 1) & ~(G - 1), skip = address - lo;
        const u64 shift = (need - skip % need) % need;
        const u32 n = static_cast<u32>((hi - lo + shift + G - 1) / G);
        const s64 g = c.Allocate(n, t.retired);
        if (g < 0) {
            if (ProfileDraws()) g_profile.cache_full.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        auto fresh = std::make_unique<E>();
        fresh->lo = lo, fresh->hi = hi, fresh->offset = static_cast<u64>(g) * G + shift;
        fresh->granule = static_cast<u32>(g), fresh->granules = n;
        fresh->gen.assign((hi - lo) / P, 0), fresh->decided.assign((hi - lo) / P, 0);
        if (CacheCheckOn()) fresh->shadow.assign(hi - lo, 0);
        if (fresh->offset + skip < min_offset) { // the arena's first bytes: a vertex binding cannot start before it
            for (u32 k = 0; k < n; ++k) c.used[static_cast<u64>(g) + k] = 0;
            if (ProfileDraws()) g_profile.cache_placed.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        e = fresh.get();
        c.entries.insert({lo, std::move(fresh)});
        c.longest = std::max(c.longest, hi - lo);
        if (ProfileDraws()) g_profile.cache_entries.fetch_add(1, std::memory_order_relaxed);
    }
    // the writes that make the entry's start-of-submission bytes wrong for some reads: earlier submissions' still in
    // flight (not in guest memory yet), and this one's so far
    if (e->sub != t.submission) {
        e->sub = t.submission, e->scanned = 0, e->written.clear();
        for (const Translator::Earlier& x : t.earlier) {
            for (const WriteBack& b : *x.writes) {
                const u64 lo = std::max(b.address, e->lo), hi = std::min(b.address + b.size, e->hi);
                if (lo < hi) e->written.push_back({lo, hi});
            }
        }
    }
    for (; e->scanned < w.size(); ++e->scanned) {
        const u64 lo = std::max(w[e->scanned].address, e->lo), hi = std::min(w[e->scanned].address + w[e->scanned].size, e->hi);
        if (lo < hi) e->written.push_back({lo, hi});
    }
    for (const auto& r : e->written) {
        if (r[0] < address + size && address < r[1]) {
            if (ProfileDraws()) g_profile.cache_written.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    // each page's first read in this submission decides: unchanged since copied, or copied again (in runs)
    const u64 a = address & ~(P - 1), b = (address + size + P - 1) & ~(P - 1);
    u64 reused = 0, copied = 0;
    for (u64 pg = a; pg < b;) {
        const size_t i = (pg - e->lo) / P;
        if (e->decided[i] == t.submission ||
            (e->decided[i] && !writetrack::ChangedSince(pg, P, e->gen[i]))) {
            e->decided[i] = t.submission, ++reused, pg += P;
            continue;
        }
        u64 end = pg + P;
        while (end < b) {
            const size_t j = (end - e->lo) / P;
            if (e->decided[j] == t.submission || (e->decided[j] && !writetrack::ChangedSince(end, P, e->gen[j]))) break;
            end += P;
        }
        writetrack::Watch(pg, end - pg); // protect, then the generation, then copy (writetrack.h)
        const u32 since = writetrack::Now();
        const auto staging = t.upload.Allocate(end - pg, 16);
        g_mem.Read(pg, staging.data, end - pg);
        t.cache_copies->push_back({staging.buffer, staging.offset, e->offset + (pg - e->lo), end - pg});
        if (!e->shadow.empty()) g_mem.Read(pg, e->shadow.data() + (pg - e->lo), end - pg);
        for (u64 q = pg; q < end; q += P) {
            const size_t j = (q - e->lo) / P;
            e->gen[j] = since, e->decided[j] = t.submission;
        }
        copied += (end - pg) / P;
        if (t.bytes) t.bytes->read_fills += end - pg;
        pg = end;
    }
    e->last_used = t.submission;
    if (ProfileDraws()) {
        g_profile.cache_reads.fetch_add(1, std::memory_order_relaxed);
        g_profile.cache_pages_reused.fetch_add(reused, std::memory_order_relaxed);
        g_profile.cache_pages_copied.fetch_add(copied, std::memory_order_relaxed);
        g_profile.cache_bytes.fetch_add(copied * P, std::memory_order_relaxed);
    }
    if (!e->shadow.empty()) { // RT_NVN_BUFFER_CACHE_CHECK: what the GPU will read against what it should see
        thread_local std::vector<u8> now;
        std::vector<GpuPart> gpu;
        now.resize(size);
        ReadAsGpuSees(t, address, size, now.data(), &gpu);
        const u64 n = g_profile.cache_checked.fetch_add(1) + 1;
        if (!gpu.empty() || std::memcmp(now.data(), e->shadow.data() + (address - e->lo), size) != 0) {
            // which pages differ, and whether write tracking saw a write to them since their copy (then the copy was
            // right when made and a CPU write came during this translation: the next submission copies again) or not
            // (a missed write: a bug)
            u64 seen = 0, missed = 0;
            for (u64 pg = a; pg < b; pg += P) {
                const u64 x = std::max(pg, address), y = std::min(pg + P, address + size);
                if (std::memcmp(now.data() + (x - address), e->shadow.data() + (x - e->lo), y - x) == 0) continue;
                const bool changed = writetrack::ChangedSince(pg, P, e->gen[(pg - e->lo) / P]);
                (changed ? seen : missed) += 1;
                static std::atomic<u32> told{0};
                if (!changed && told.fetch_add(1) < 12) {
                    u64 k = x;
                    while (k < y && now[k - address] == e->shadow[k - e->lo]) ++k;
                    Log("nvn: buffer cache check: MISSED page 0x%llx (entry 0x%llx-0x%llx, copied at generation %u in submission "
                        "%llu, now submission %llu; writetrack: %s); first difference at +0x%llx: copy %02x, guest %02x",
                        (unsigned long long)pg, (unsigned long long)e->lo, (unsigned long long)e->hi, e->gen[(pg - e->lo) / P],
                        (unsigned long long)e->decided[(pg - e->lo) / P], (unsigned long long)t.submission,
                        writetrack::Describe(pg).c_str(), (unsigned long long)(k - pg), e->shadow[k - e->lo], now[k - address]);
                }
            }
            const u64 d = g_profile.cache_differ.fetch_add(1) + 1;
            if (missed) g_profile.cache_missed.fetch_add(missed);
            if (d <= 20 || d % 1000 == 0 || missed) {
                Log("nvn: buffer cache check: a served read of 0x%llx+0x%llx differs from guest memory as the GPU sees it "
                    "(#%llu; %zu GPU parts; pages differing: %llu written during this translation (seen), %llu MISSED)",
                    (unsigned long long)address, (unsigned long long)size, (unsigned long long)d, gpu.size(),
                    (unsigned long long)seen, (unsigned long long)missed);
            }
        }
        if (n % 100000 == 0) {
            Log("nvn: buffer cache check: %llu served reads compared, %llu differ", (unsigned long long)n,
                (unsigned long long)g_profile.cache_differ.load());
        }
    }
    out = {*c.arena, e->offset + (address - e->lo), nullptr};
    // RT_NVN_BUFFER_CACHE_CHECK on the GPU: every 16th served read (up to 64 MB: the faces' vertex reads may be large) copied out of the cache at this
    // point of the commands, compared at retirement with what it should be (queue.cpp)
    static std::atomic<u64> sampled{0};
    if (!e->shadow.empty() && t.cache_verify && t.download && size <= (u64{64} << 20) && sampled.fetch_add(1) % 16 == 0) {
        std::vector<u8> expected(size);
        std::vector<GpuPart> gpu;
        ReadAsGpuSees(t, address, size, expected.data(), &gpu);
        if (gpu.empty()) {
            EndOpenPass(t.cmd);
            const auto dst = t.download->Allocate(size, 16);
            t.cmd.CopyBuffer(*c.arena, dst.buffer, VkBufferCopy{out.offset, dst.offset, size});
            t.cache_verify->push_back({address, static_cast<u32>(t_phase), dst.data, std::move(expected)});
        }
    }
    return true;
}

/// A copy of guest memory for the GPU (FillAsGpuSees), shared by the draws of a submission; at least
/// `min_offset` into its buffer (UploadRing::Allocate).
// A copy stays valid while no write recorded after it (GPU writes, CPU-known data the GPU writes) touches its
// bytes; only the writes added since its last check are looked at. 2026-10-23: the plaza copied 18 MB per frame
// with a copy dropped at any write anywhere, 7 MB of it distinct (mostly texel buffers several draws read).
// `need`: the alignment Vulkan requires of the binding's offset (0: `alignment`; vertex buffers have none, index
// buffers their index size): a page copy (PageCopy) keeps the guest address's own alignment instead of `alignment`.
UploadRing::Slice Snapshot(Translator& t, u64 address, u64 size, u64 alignment, u64 min_offset = 0, u64 need = 0) {
    if (!need) need = alignment;
    const std::array<u64, 2> key{address, size};
    auto it = t.snapshots.find(key);
    if (it != t.snapshots.end() && ProfileDraws() &&
        (it->second.slice.offset % need != 0 || it->second.slice.offset < min_offset)) {
        g_profile.snap_placed.fetch_add(1, std::memory_order_relaxed);
    }
    if (it != t.snapshots.end() && it->second.slice.offset % need == 0 && it->second.slice.offset >= min_offset) {
        const std::vector<WriteBack>& w = *t.writebacks;
        bool stale = false;
        for (size_t i = it->second.writes; i < w.size() && !stale; ++i) {
            stale = w[i].address < address + size && address < w[i].address + w[i].size;
        }
        if (!stale) {
            it->second.writes = w.size();
            // RT_NVN_SNAPSHOT_CHECK=1 (test aid): a reused copy against guest memory read again, where no GPU part
            // is involved (those bytes are on the GPU only); differences are counted and logged
            static const bool check = std::getenv("RT_NVN_SNAPSHOT_CHECK") != nullptr;
            if (check && it->second.slice.data) { // (a cached read has no CPU copy: RT_NVN_BUFFER_CACHE_CHECK)
                static std::atomic<u64> checked{0}, differ{0};
                thread_local std::vector<u8> now;
                std::vector<GpuPart> gpu;
                now.resize(size);
                ReadAsGpuSees(t, address, size, now.data(), &gpu);
                if (gpu.empty()) {
                    const u64 n = checked.fetch_add(1) + 1;
                    if (std::memcmp(now.data(), it->second.slice.data, size) != 0) {
                        Log("nvn: snapshot check: a reused copy of 0x%llx+0x%llx differs from guest memory (#%llu)",
                            (unsigned long long)address, (unsigned long long)size, (unsigned long long)differ.fetch_add(1) + 1);
                    }
                    if (n % 100000 == 0) {
                        Log("nvn: snapshot check: %llu reused copies checked, %llu differ", (unsigned long long)n,
                            (unsigned long long)differ.load());
                    }
                }
            }
            if (ProfileDraws()) g_profile.snap_hits.fetch_add(1, std::memory_order_relaxed);
            NoteRead(address, size, it->second.slice.data, nullptr, true);
            return it->second.slice;
        }
        if (ProfileDraws()) g_profile.snap_stale.fetch_add(1, std::memory_order_relaxed);
    }
    if (ProfileDraws()) g_profile.snap_misses.fetch_add(1, std::memory_order_relaxed);
    if (UploadRing::Slice cached{}; CachedRead(t, address, size, need, min_offset, cached)) {
        if (RangeStatsOn()) RangeStatsNote(address, size, false);
        t.snapshots[key] = {cached, std::min(t.writebacks->size(), t.pending_from)};
        return cached;
    }
    UploadRing::Slice paged{};
    if (PageCopyOn() && PageCopy(t, address, size, need, min_offset, paged)) {
        // RT_NVN_SNAPSHOT_CHECK=1: the span's bytes for this read against guest memory (where no GPU part is involved)
        static const bool check = std::getenv("RT_NVN_SNAPSHOT_CHECK") != nullptr;
        if (check && paged.data) {
            static std::atomic<u64> checked{0}, differ{0};
            thread_local std::vector<u8> now;
            std::vector<GpuPart> gpu;
            now.resize(size);
            ReadAsGpuSees(t, address, size, now.data(), &gpu);
            if (gpu.empty()) {
                const u64 n = checked.fetch_add(1) + 1;
                if (std::memcmp(now.data(), paged.data, size) != 0) {
                    Log("nvn: page copy check: the span's copy of 0x%llx+0x%llx differs from guest memory (#%llu)",
                        (unsigned long long)address, (unsigned long long)size, (unsigned long long)differ.fetch_add(1) + 1);
                }
                if (n % 100000 == 0) {
                    Log("nvn: page copy check: %llu reads checked, %llu differ", (unsigned long long)n, (unsigned long long)differ.load());
                }
            }
        }
        NoteRead(address, size, paged.data, nullptr, true);
        // RT_NVN_PAGE_COPY_CHECK=1 (test aid): every 8th page-copied read also copied on its own as before (a slice
        // of its own, FillAsGpuSees), both copied out at this point of the commands and compared at retirement
        // (queue.cpp): the faces' bug was in the page copy, 2026-10-24, where no check looked at GPU-written parts
        static const bool page_check = std::getenv("RT_NVN_PAGE_COPY_CHECK") != nullptr;
        static std::atomic<u64> page_sampled{0};
        if (page_check && t.cache_verify && t.download && size <= (u64{16} << 20) && page_sampled.fetch_add(1) % 8 == 0) {
            const auto own = t.upload.Allocate(size, 16);
            bool gpu = false, made = true;
            try {
                gpu = FillAsGpuSees(t, address, size, own);
            } catch (const NotYet&) {
                made = false;
            }
            if (made) {
                EndOpenPass(t.cmd);
                const auto a = t.download->Allocate(size, 16), b = t.download->Allocate(size, 16);
                t.cmd.CopyBuffer(paged.buffer, a.buffer, VkBufferCopy{paged.offset, a.offset, size});
                t.cmd.CopyBuffer(own.buffer, b.buffer, VkBufferCopy{own.offset, b.offset, size});
                CacheVerify v{address, static_cast<u32>(t_phase), a.data, std::vector<u8>(size)};
                v.ref = b.data, v.gpu_parts = gpu, v.note = t_page_note;
                t.cache_verify->push_back(std::move(v));
            }
        }
        if (RangeStatsOn()) RangeStatsNote(address, size, false);
        t.snapshots[key] = {paged, std::min(t.writebacks->size(), t.pending_from)};
        return paged;
    }
    const auto slice = t.upload.Allocate(size, alignment, min_offset);
    const bool gpu = FillAsGpuSees(t, address, size, slice);
    if (RangeStatsOn()) RangeStatsNote(address, size, gpu);
    if (t.bytes) t.bytes->read_fills += size;
    t.snapshots[key] = {slice, std::min(t.writebacks->size(), t.pending_from)}; // (pending writes: as PageCopy's fills)
    return slice;
}

/// The lowest and highest of `n` indices of NVNindexType `type` (0 u8, 1 u16, 2 u32), as the GPU sees them; false
/// when a part of them only the GPU has (then the caller copies the vertex buffers whole).
bool IndexRange(Translator& t, u64 address, u32 type, u64 n, u32& lo, u32& hi) {
    thread_local std::vector<u8> buf;
    buf.resize(n << type);
    std::vector<GpuPart> gpu;
    ReadAsGpuSees(t, address, buf.size(), buf.data(), &gpu);
    if (!gpu.empty() || !n) return false;
    auto scan = [&](auto* v) {
        u32 a = v[0], b = v[0];
        for (u64 i = 1; i < n; ++i) a = std::min<u32>(a, v[i]), b = std::max<u32>(b, v[i]);
        lo = a, hi = b;
    };
    if (type == 0) scan(buf.data());
    else if (type == 1) scan(reinterpret_cast<const u16*>(buf.data()));
    else scan(reinterpret_cast<const u32*>(buf.data()));
    return true;
}

VkImageViewType ViewType(Shader::TextureType type) {
    switch (type) {
    case Shader::TextureType::Color1D: return VK_IMAGE_VIEW_TYPE_1D;
    case Shader::TextureType::ColorArray1D: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case Shader::TextureType::Color2D:
    case Shader::TextureType::Color2DRect: return VK_IMAGE_VIEW_TYPE_2D;
    case Shader::TextureType::ColorArray2D: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case Shader::TextureType::Color3D: return VK_IMAGE_VIEW_TYPE_3D;
    case Shader::TextureType::ColorCube: return VK_IMAGE_VIEW_TYPE_CUBE;
    case Shader::TextureType::ColorArrayCube: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    case Shader::TextureType::Buffer: break;
    }
    throw NotYet("buffer textures");
}

/// Constant buffer `bank` of a stage for a draw: `range` bytes (zero past what the source holds).
/// During a draw with GPU-written parameters (Draw's gpu_params): the graphics stages' c0 copies that reach the
/// base vertex and base instance words, for Draw to copy the GPU's values over them.
thread_local std::vector<VkDescriptorBufferInfo>* t_c0_copies = nullptr;

VkDescriptorBufferInfo ConstantBuffer(Translator& t, u32 nvn_stage, const ProgramStage& program, const Shader::Info& info,
                                      u32 bank) {
    const RenderState& rs = *t.render;
    u32 range = info.constant_buffer_used_sizes[bank];
    range = range ? std::min<u32>((range + 15) & ~15u, 0x10000) : 0x10000; // 0: indexed at run time
    const u64 align = Vk().device->GetUniformBufferAlignment();
    UploadRing::Slice slice{};
    auto copy = [&](const u8* src, u64 n) {
        slice = t.upload.Allocate(range, align);
        std::memset(slice.data, 0, range);
        std::memcpy(slice.data, src, std::min<u64>(n, range));
        ProfileBytes(range);
    };
    if (bank == 0 && nvn_stage == 5) {
        const auto c0 = ComputeC0(rs);
        copy(c0.data(), c0.size());
        if (t_c0_copies && range >= kComputeC0Groups + 12) t_c0_copies->push_back({slice.buffer, slice.offset, range});
    } else if (bank == 0) {
        copy(rs.c0.data(), kC0Size);
        if (t_c0_copies && range >= kC0BaseInstance + 4) t_c0_copies->push_back({slice.buffer, slice.offset, range});
    } else if (bank == 2) {
        copy(rs.c2[C2Index(nvn_stage)].data(), kC2Size);
    } else if (bank == 1) { // the compiler constants, in the stage's code memory
        std::vector<u8> v(std::min<u32>(program.constants_size, range));
        g_mem.Read(program.code + program.constants_offset, v.data(), v.size());
        copy(v.data(), v.size());
    } else {
        const auto [address, size] = rs.bindings[nvn_stage].uniform_buffers.at(bank - 3);
        if (!address) throw NotYet("a shader reading an unbound uniform buffer");
        if (size >= range) {
            slice = Snapshot(t, address, range, align);
        } else { // smaller than the shader reads: zero past it
            slice = t.upload.Allocate(range, align);
            std::memset(slice.data, 0, range);
            FillAsGpuSees(t, address, size, slice);
        }
    }
    return {slice.buffer, slice.offset, range};
}

/// A linear texture (flag 0x10: plain rows of `stride` bytes in guest memory, which the CPU writes — a
/// video decoder's planes, for example) is guest memory first: uploaded into its image as the GPU sees the
/// memory, once per submission, before the draw or dispatch that samples it.
void UploadLinear(Translator& t, const TextureState& tex, TextureImage& img) {
    const TextureBuilder& b = tex.settings;
    if (!(b.flags & 0x10) || (b.flags & 0x40)) return; // not linear, or a linear render target (the GPU's)
    if (!t.linear_uploads.insert(&img).second) return;
    const FormatInfo* f = FindFormat(b.format);
    if (!f || f->block_w != 1 || b.stride <= 0 || b.stride % f->bytes) throw NotYet("a linear texture of this format or stride");
    if (b.target != 1 || b.levels > 1) throw NotYet("linear textures other than 2D with one level");
    const u64 size = u64(b.stride) * (b.height - 1) + u64{b.width} * f->bytes;
    const auto slice = t.upload.Allocate(size, 16);
    FillAsGpuSees(t, tex.address, size, slice);
    const VkBufferImageCopy region{
        .bufferOffset = slice.offset,
        .bufferRowLength = static_cast<u32>(b.stride) / f->bytes,
        .bufferImageHeight = 0,
        .imageSubresource = {img.aspect, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {b.width, b.height, 1},
    };
    EndOpenPass(t.cmd);
    t.cmd.CopyBufferToImage(slice.buffer, *img.image, VK_IMAGE_LAYOUT_GENERAL, region);
    FullBarrier(t.cmd);
    static std::atomic<u32> told{0};
    if (told.fetch_add(1) < 4) {
        Log("nvn: linear texture %ux%u format 0x%x (stride %d) uploaded from guest memory", b.width, b.height,
            b.format, b.stride);
    }
}

/// Memory a shader writes (a storage buffer or a buffer texture at `address`): the game's data as the GPU sees
/// it in the download ring, and a write-back from there to guest memory at retirement. Returns the slice to
/// bind and sets `*ring` to the ring it is in. A compute shader works on a device-memory copy instead, copied
/// into the download slice after the dispatch (Dispatch, t.after_dispatch): its atomics then run in video
/// memory, not across the bus in host memory, where the 3090 Ti lost some of the face morphs' concurrent adds
/// on the results screen (2026-10-21). RT_NVN_HOST_WRITES=1: in the download ring as before (A/B).
UploadRing::Slice ShaderWrite(Translator& t, u64 address, u64 size, u64 align, u32 nvn_stage, WriteBack::Origin origin,
                              UploadRing*& ring) {
    static const bool host = [] {
        const char* e = std::getenv("RT_NVN_HOST_WRITES");
        return e && *e && *e != '0';
    }();
    std::optional<PartTimer> setup;
    setup.emplace(kWriteSetup);
    const auto slice = t.download->Allocate(size, align);
    setup.reset();
    FillAsGpuSees(t, address, size, slice);
    setup.emplace(kWriteSetup);
    if (t.bytes) t.bytes->write_fills += size, ++t.bytes->writes, t.bytes->largest_write = std::max(t.bytes->largest_write, size);
    t.writebacks->push_back({WriteBack::Kind::kRing, address, size, slice.data, {}, 0, slice.buffer, slice.offset});
    t.writebacks->back().origin = origin;
    t.writebacks->back().stage = static_cast<u8>(nvn_stage);
    ring = t.download;
    if (host || nvn_stage != 5 || !t.device) return slice;
    const auto device = t.device->Allocate(size, align);
    EndOpenPass(t.cmd);
    t.cmd.CopyBuffer(slice.buffer, device.buffer, VkBufferCopy{slice.offset, device.offset, size});
    if (t.bytes) t.bytes->to_device += size;
    FullBarrier(t.cmd);
    t.after_dispatch.push_back({device, slice});
    t.after_dispatch_sizes.push_back(size);
    ring = t.device;
    return device;
}

/// The Vulkan format and element size a shader declares for an image buffer (drippu's ImageFormat, from the
/// instruction: typed surface loads, stores and atomics), or {VK_FORMAT_UNDEFINED, 0} for Typeless.
std::pair<VkFormat, u32> DeclaredFormat(Shader::ImageFormat f) {
    switch (f) {
    case Shader::ImageFormat::Typeless: break;
    case Shader::ImageFormat::R8_UINT: return {VK_FORMAT_R8_UINT, 1};
    case Shader::ImageFormat::R8_SINT: return {VK_FORMAT_R8_SINT, 1};
    case Shader::ImageFormat::R16_UINT: return {VK_FORMAT_R16_UINT, 2};
    case Shader::ImageFormat::R16_SINT: return {VK_FORMAT_R16_SINT, 2};
    case Shader::ImageFormat::R32_UINT: return {VK_FORMAT_R32_UINT, 4};
    case Shader::ImageFormat::R32G32_UINT: return {VK_FORMAT_R32G32_UINT, 8};
    case Shader::ImageFormat::R32G32B32A32_UINT: return {VK_FORMAT_R32G32B32A32_UINT, 16};
    }
    return {VK_FORMAT_UNDEFINED, 0};
}

/// A texel-buffer view of texture `id` (a buffer texture: `width` elements of its format at its address), its
/// data copied as the GPU sees it; a `written` one goes through the download ring back to guest memory.
/// `declared`: an image buffer's format as the shader declares it (Typeless for texture buffers and untyped
/// image access). A typed declaration sets the view's format over the same bytes, as yuzu's buffer cache
/// does (pipeline_helper.h PixelFormatFromImageFormat): Vulkan wants the view to match a declared format,
/// and atomics need it (R32_UINT; 2026-10-21: the results screen's face morphs add into buffer textures).
VkBufferView TexelBuffer(Translator& t, u32 id, bool written, u32 nvn_stage,
                         Shader::ImageFormat declared = Shader::ImageFormat::Typeless) {
    const RenderState& rs = *t.render;
    std::optional<PartTimer> lookup;
    lookup.emplace(kLookup);
    const auto e = TexturePoolEntry(rs.texture_pool, id);
    if (!e) throw NotYet("a buffer-texture handle naming no registered texture");
    const TextureBuilder& b = e->texture.settings;
    if (b.target != 10) throw NotYet("a buffer-texture binding of a texture that is no buffer texture");
    const FormatInfo* f = FindFormat(b.format);
    VkFormat format = VulkanFormat(b.format);
    if (!f || format == VK_FORMAT_UNDEFINED) throw NotYet("a buffer texture of a format without a Vulkan format");
    u64 size = u64{b.width} * f->bytes;
    if (const auto [typed, bytes] = DeclaredFormat(declared); typed != VK_FORMAT_UNDEFINED && typed != format) {
        static std::mutex mu;
        static std::set<std::pair<VkFormat, VkFormat>> told;
        if (std::lock_guard lock(mu); told.insert({format, typed}).second) {
            Log("nvn: an image buffer the shader declares as Vulkan format %d over a buffer texture of format %d "
                "(NVN 0x%x): viewed as the declared format", typed, format, b.format);
        }
        format = typed;
        size -= size % bytes;
    }
    const VkFormatFeatureFlags need = written ? VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT : VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT;
    if (!(Vk().device->GetPhysical().GetFormatProperties(format).bufferFeatures & need)) {
        throw NotYet("a buffer texture of a format the device cannot use as a texel buffer");
    }
    static const u64 align = std::max<u64>(Vk().device->GetPhysical().GetProperties().limits.minTexelBufferOffsetAlignment, 16);
    lookup.reset();
    if (written) {
        UploadRing* ring = nullptr;
        const auto slice = ShaderWrite(t, e->texture.address, size, align, nvn_stage, WriteBack::Origin::kBufferTexture, ring);
        PartTimer view(kView);
        return ring->TexelView(slice, size, format);
    }
    const auto slice = Snapshot(t, e->texture.address, size, align);
    PartTimer view(kView);
    return t.upload.TexelView(slice, size, format);
}

/// The descriptors of one stage, appended in DescriptorLayoutBuilder's order.
void AddDescriptors(Translator& t, u32 nvn_stage, const ProgramStage& program, const Shader::Info& info,
                    std::vector<Vulkan::DescriptorUpdateEntry>& entries, RescaleBits& bits) {
    const RenderState& rs = *t.render;
    std::optional<PhaseTimer> phase;
    phase.emplace(kCbuf);
    for (const auto& cb : info.constant_buffer_descriptors) {
        for (u32 k = 0; k < cb.count; ++k) entries.emplace_back(ConstantBuffer(t, nvn_stage, program, info, cb.index + k));
    }
    phase.reset(), phase.emplace(kStorage);
    for (const auto& sb : info.storage_buffers_descriptors) {
        for (u32 k = 0; k < sb.count; ++k) {
            const u32 off = sb.cbuf_offset + 16 * k;
            const u64 address = CbufWord(rs, nvn_stage, sb.cbuf_index, off) |
                                u64{CbufWord(rs, nvn_stage, sb.cbuf_index, off + 4)} << 32;
            const u32 size = CbufWord(rs, nvn_stage, sb.cbuf_index, off + 8);
            if (!address || !size) throw NotYet("an unbound storage buffer");
            const u64 align = Vk().device->GetStorageBufferAlignment();
            if (sb.is_written) {
                UploadRing* ring = nullptr;
                const auto slice = ShaderWrite(t, address, size, align, nvn_stage, WriteBack::Origin::kStorageBuffer, ring);
                entries.emplace_back(VkDescriptorBufferInfo{slice.buffer, slice.offset, size});
            } else {
                const auto slice = Snapshot(t, address, size, align);
                entries.emplace_back(VkDescriptorBufferInfo{slice.buffer, slice.offset, size});
            }
        }
    }
    phase.reset(), phase.emplace(kTexel);
    for (const auto& tb : info.texture_buffer_descriptors) {
        if (tb.count != 1) throw NotYet("texture buffer arrays");
        u32 raw = CbufWord(rs, nvn_stage, tb.cbuf_index, tb.cbuf_offset) << tb.shift_left;
        if (tb.has_secondary) {
            raw |= CbufWord(rs, nvn_stage, tb.secondary_cbuf_index, tb.secondary_cbuf_offset) << tb.secondary_shift_left;
        }
        entries.emplace_back(TexelBuffer(t, raw & 0xFFFFF, false, nvn_stage));
    }
    for (const auto& ib : info.image_buffer_descriptors) {
        if (ib.count != 1) throw NotYet("image buffer arrays");
        entries.emplace_back(TexelBuffer(t, CbufWord(rs, nvn_stage, ib.cbuf_index, ib.cbuf_offset) & 0xFFFFF, ib.is_written,
                                         nvn_stage, ib.format));
    }
    phase.reset(), phase.emplace(kTextures);
    const RescaleRange range = RescaleRangeOfNvn(nvn_stage);
    u32 texture_index = 0;
    for (const auto& td : info.texture_descriptors) {
        if (td.count != 1) throw NotYet("texture arrays");
        if (td.is_multisample) throw NotYet("multisample textures");
        // the handle as drippu combines it: a separate texture's id | a separate sampler's id << 20
        u32 raw = CbufWord(rs, nvn_stage, td.cbuf_index, td.cbuf_offset) << td.shift_left;
        if (td.has_secondary) {
            raw |= CbufWord(rs, nvn_stage, td.secondary_cbuf_index, td.secondary_cbuf_offset) << td.secondary_shift_left;
        }
        const u32 tex = raw & 0xFFFFF, smp = raw >> 20;
        const auto e = TexturePoolEntry(rs.texture_pool, tex);
        if (!e) throw NotYet("a texture handle naming no registered texture");
        const auto img = ImageFor(e->texture, t.cmd);
        t.keep.push_back(img);
        UploadLinear(t, e->texture, *img);
        const VkImageView view = SampledView(*img, e->texture, e->view, ViewType(td.type));
        NoteImageUse(e->texture, ImageUse::kSampled);
        bits.Note(range, false, texture_index++, img->Scaled());
        entries.emplace_back(VkDescriptorImageInfo{SamplerFor(rs.sampler_pool, smp), view, VK_IMAGE_LAYOUT_GENERAL});
    }
    phase.reset(), phase.emplace(kImages);
    u32 image_index = 0;
    for (const auto& id : info.image_descriptors) {
        if (id.count != 1) throw NotYet("image arrays");
        const u32 raw = CbufWord(rs, nvn_stage, id.cbuf_index, id.cbuf_offset); // GetImageHandle: the id
        const auto e = TexturePoolEntry(rs.texture_pool, raw & 0xFFFFF);
        if (!e) throw NotYet("an image handle naming no registered image");
        const auto img = ImageFor(e->texture, t.cmd);
        t.keep.push_back(img);
        if (!img->storage) throw NotYet("storage images of a format the device cannot store to");
        const VkImageView view = StorageView(*img, e->texture, e->view, ViewType(id.type));
        NoteImageUse(e->texture, ImageUse::kStorage);
        bits.Note(range, true, image_index++, img->Scaled());
        if (img->Scaled() && id.is_written) { // the rescaling pass scales reads, not writes (as drippu's)
            static std::atomic<bool> told{false};
            if (!told.exchange(true)) Log("resolution: a scaled image is bound as a written storage image: its writes are not scaled");
        }
        entries.emplace_back(VkDescriptorImageInfo{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL});
    }
}

/// Binds `entries` for pipeline `p` (Graphics- or ComputePipeline): pushed, or as a set from the submission's
/// DescriptorArena when the pipeline has more descriptors than the device pushes.
template <typename P>
void BindDescriptors(Translator& t, VkPipelineBindPoint point, const P& p,
                     const std::vector<Vulkan::DescriptorUpdateEntry>& entries) {
    if (entries.empty()) return;
    if (p.push) {
        t.cmd.PushDescriptorSetWithTemplateKHR(*p.update, *p.layout, 0, entries.data());
        return;
    }
    if (!t.sets) throw NotYet("a descriptor set outside a submission");
    const VkDescriptorSet set = t.sets->Allocate(*p.set_layout);
    Vk().device->GetLogical().UpdateDescriptorSet(set, *p.update, entries.data());
    t.cmd.BindDescriptorSets(point, *p.layout, 0, set, {});
}

/// Binds the descriptors of a pipeline of stages built alone (pipeline.h StageBuild): each stage's own set,
/// pushed (a fragment stage's, when it fits) or from the submission's DescriptorArena.
void BindStageDescriptors(Translator& t, const GraphicsPipeline& p,
                          const std::vector<std::vector<Vulkan::DescriptorUpdateEntry>>& entries) {
    for (size_t i = 0; i < p.stages.size(); ++i) {
        const GraphicsPipeline::Stage& st = p.stages[i];
        if (entries[i].empty()) continue;
        if (st.built->push) {
            t.cmd.PushDescriptorSetWithTemplateKHR(*st.built->push_update, *p.layout, st.set, entries[i].data());
            continue;
        }
        if (!t.sets) throw NotYet("a descriptor set outside a submission");
        const VkDescriptorSet set = t.sets->Allocate(*st.built->set_layout);
        Vk().device->GetLogical().UpdateDescriptorSet(set, *st.built->set_update, entries[i].data());
        t.cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *p.layout, st.set, set, {});
    }
}

std::atomic<u64> g_drawn{0}, g_skipped{0}, g_waiting{0}; // g_waiting: skipped while their pipeline was built

/// `gpu_params`: the parameters of an indirect draw that only the GPU has (their copy, TranslateDrawRecord):
/// drawn with vkCmdDraw(Indexed)Indirect from it, the base vertex and base instance copied into c0 on the GPU,
/// the vertex buffers copied whole; `r` then names the index buffer's start and, in `count`, how many indices
/// to copy from it.
/// Translator::pending_from for the scope of a draw or dispatch (also when it stops at NotYet).
struct PendingWrites {
    Translator& t;
    explicit PendingWrites(Translator& tr) : t(tr) { t.pending_from = t.writebacks->size(); }
    ~PendingWrites() { t.pending_from = SIZE_MAX; }
};

void Draw(Translator& t, const DrawRecord& r, const UploadRing::Slice* gpu_params = nullptr) {
    PendingWrites pending(t);
    RenderState& rs = *t.render;
    const auto& cmd = t.cmd;
    std::optional<PhaseTimer> phase;
    phase.emplace(kTargets);
    const TargetSet targets = BindTargets(t);
    phase.reset(), phase.emplace(kPipeline);
    const GraphicsPipeline& p = PipelineFor(rs, r.mode, targets);
    const ClipModes clip = DeviceClipModes(rs.device);
    phase.reset();

    // c0's draw parameters, as NVN writes them (the base vertex / first vertex and base instance)
    const s32 draw_words[2] = {r.first, r.base_instance};
    std::memcpy(rs.c0.data() + kC0BaseVertex, draw_words, 8);

    // descriptors, in DescriptorLayoutBuilder's order: one list, or one per stage (stages built alone)
    const bool alone = !p.stages.empty() && p.stages.front().built;
    std::vector<Vulkan::DescriptorUpdateEntry> entries;
    std::vector<std::vector<Vulkan::DescriptorUpdateEntry>> stage_entries(alone ? p.stages.size() : 0);
    std::vector<VkDescriptorBufferInfo> c0_copies;
    RescaleBits rescale; // the stages' scaled bindings (each in its own bits)
    {
        struct Capture { // t_c0_copies for this scope, also when a descriptor throws NotYet
            explicit Capture(std::vector<VkDescriptorBufferInfo>* v) { t_c0_copies = v; }
            ~Capture() { t_c0_copies = nullptr; }
        } capture(gpu_params ? &c0_copies : nullptr);
        for (size_t i = 0; i < p.stages.size(); ++i) {
            const GraphicsPipeline::Stage& st = p.stages[i];
            AddDescriptors(t, st.nvn_stage, st.program, st.info, alone ? stage_entries[i] : entries, rescale);
        }
    }
    if (!c0_copies.empty()) { // base vertex (arrays: first vertex), base instance: words 3-4 (arrays 2-3) of the parameters
        EndOpenPass(cmd);
        const u64 from = gpu_params->offset + (r.indexed ? 12 : 8);
        for (const auto& c : c0_copies) cmd.CopyBuffer(gpu_params->buffer, c.buffer, VkBufferCopy{from, c.offset + kC0BaseVertex, 8});
        FullBarrier(cmd);
    }

    // vertex streams read by the attributes the vertex shader loads (pipeline.cpp), and the indices
    const u32 streams = p.vertex_streams;
    // Only the part of each stream the draw reads is copied: vertices first .. first + count - 1 (arrays) or
    // the indices' lowest .. highest plus the base vertex, instances base .. base + instances - 1 (divisor 1),
    // each to its attributes' last byte (offset + 16 at most). The copy is bound where the whole buffer's copy
    // would start, so the vertex addresses don't change. Anything else (indices only the GPU has, negative
    // vertices, a range past the binding, a part starting far into a large buffer) copies the binding whole.
    // RT_NVN_WHOLE_VERTEX=1 copies every binding whole (as before 2026-10-19; for A/B checks).
    static const bool whole_vertex = [] {
        const char* e = std::getenv("RT_NVN_WHOLE_VERTEX");
        const bool on = e && *e && *e != '0';
        if (on) Log("nvn: vertex buffers copied whole (RT_NVN_WHOLE_VERTEX)");
        return on;
    }();
    phase.emplace(kIndex);
    s64 v_lo = r.first, v_hi = static_cast<s64>(r.first) + r.count - 1;
    bool v_known = r.count > 0 && !whole_vertex && !gpu_params;
    if (v_known && r.indexed) {
        u32 lo = 0, hi = 0;
        v_known = r.indexed <= 3 && IndexRange(t, r.index_address, r.indexed - 1, static_cast<u64>(r.count), lo, hi);
        v_lo += lo, v_hi = static_cast<s64>(r.first) + hi;
    }
    phase.reset(), phase.emplace(kVertex);
    const s64 instances_n = std::max(r.instances, 0);
    std::vector<std::pair<u32, UploadRing::Slice>> vbufs;
    for (u32 s = 0; s < 16; ++s) {
        if (!(streams & (1u << s))) continue;
        const auto [address, size] = rs.vertex_buffers[s];
        if (!address || !size) throw NotYet("an unbound vertex buffer");
        const auto [stride, divisor] = rs.vertex_streams[s];
        const s64 lo = divisor ? r.base_instance : v_lo;
        const s64 hi = divisor ? static_cast<s64>(r.base_instance) + instances_n - 1 : v_hi;
        u64 begin = 0, end = size;
        if (!whole_vertex && ((divisor && !gpu_params) || v_known) && lo >= 0 && hi >= lo) {
            u64 span = 0;
            for (u32 i = 0; i < rs.vertex_attrib_count; ++i) {
                const u32 bits = rs.vertex_attribs[i];
                if ((bits & 0xFF) && ((bits >> 8) & 0x1F) == s) span = std::max<u64>(span, (bits >> 13) + 16);
            }
            const u64 b = static_cast<u64>(lo) * stride, e = static_cast<u64>(hi) * stride + span;
            if (b < size) begin = b <= (1ULL << 20) ? b : 0, end = std::min(e, size);
        }
        if (ProfileDraws()) {
            g_profile.vertex_bound.fetch_add(size, std::memory_order_relaxed);
            g_profile.vertex_streams.fetch_add(1, std::memory_order_relaxed);
            if (begin || end < size) g_profile.vertex_ranged.fetch_add(1, std::memory_order_relaxed);
        }
        UploadRing::Slice slice = Snapshot(t, address + begin, end - begin, 16, begin, 1);
        slice.offset -= begin;
        vbufs.push_back({s, slice});
    }
    phase.reset(), phase.emplace(kIndex);
    UploadRing::Slice ibuf{};
    VkIndexType itype = VK_INDEX_TYPE_UINT32;
    if (r.indexed) {
        const u32 type = r.indexed - 1; // NVNindexType: 0 u8, 1 u16, 2 u32
        if (type > 2 || r.count < 0) throw NotYet("index type");
        const u64 n = static_cast<u64>(r.count);
        if (type == 0 && !Vk().device->IsExtIndexTypeUint8Supported()) { // widened to 16 bits
            std::vector<u8> v(n);
            ReadAsGpuSees(t, r.index_address, n, v.data());
            ibuf = t.upload.Allocate(2 * n, 4);
            for (u64 i = 0; i < n; ++i) reinterpret_cast<u16*>(ibuf.data)[i] = v[i];
            itype = VK_INDEX_TYPE_UINT16;
        } else {
            ibuf = Snapshot(t, r.index_address, n << type, 4, 0, u64{1} << type);
            itype = type == 0 ? VK_INDEX_TYPE_UINT8_EXT : type == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        }
    }

    phase.reset(), phase.emplace(kRecord);
    // the draw, in the pass the previous draw left open if the targets are the same
    const VkRenderPassBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = targets.pass,
        .framebuffer = targets.framebuffer,
        .renderArea = {{0, 0}, targets.extent},
        .clearValueCount = 0,
        .pClearValues = nullptr,
    };
    BeginDrawPass(cmd, begin);
    const VkPipeline optimised = p.optimised.load(std::memory_order_acquire); // RT_NVN_GPL_OPTIMISE's re-link, once made
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, optimised ? optimised : *p.pipeline);
    if (p.linked && t.dynamic_of != &p) { // the state its libraries leave to the draw (pipeline.cpp), when it changes
        t.dynamic_of = &p;
        const GraphicsPipeline::Dynamic& d = p.dynamic;
        cmd.SetCullModeEXT(d.cull);
        cmd.SetFrontFaceEXT(d.front);
        cmd.SetDepthBiasEnableEXT(d.bias);
        cmd.SetDepthBias(d.bias_constant, d.bias_clamp, d.bias_slope);
        cmd.SetDepthTestEnableEXT(d.depth_test);
        cmd.SetDepthWriteEnableEXT(d.depth_write);
        cmd.SetDepthCompareOpEXT(d.depth_compare);
        cmd.SetDepthBoundsTestEnableEXT(d.depth_bounds);
        cmd.SetStencilTestEnableEXT(d.stencil_test);
        cmd.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_BIT, d.front_ops.failOp, d.front_ops.passOp, d.front_ops.depthFailOp,
                            d.front_ops.compareOp);
        cmd.SetStencilOpEXT(VK_STENCIL_FACE_BACK_BIT, d.back_ops.failOp, d.back_ops.passOp, d.back_ops.depthFailOp,
                            d.back_ops.compareOp);
        if (!d.depth_bounds) cmd.SetDepthBounds(0.0f, 1.0f); // dynamic in the library: set even when unused
    }
    if (alone) {
        BindStageDescriptors(t, p, stage_entries);
    } else {
        BindDescriptors(t, VK_PIPELINE_BIND_POINT_GRAPHICS, p, entries);
    }
    using namespace Shader::Backend::SPIRV;
    // scaled targets (all of the draw's, BindTargets) - fragment coordinates back in the
    // game's units (down_factor), viewport and scissor in the targets' own; the render area (drippu's position pass
    // turns pixel positions into clip space by it) stays the game's size
    const bool scaled = targets.scale_n != 4;
    const float up = static_cast<float>(targets.scale_n) / 4.0f;
    const RescaleBits::Words rescaling = RescalingWords(rescale.scaled);
    const float down_factor = scaled ? 1.0f / up : 1.0f;
    cmd.PushConstants(*p.layout, VK_SHADER_STAGE_ALL_GRAPHICS, RESCALING_LAYOUT_WORDS_OFFSET, sizeof rescaling, rescaling.data());
    cmd.PushConstants(*p.layout, VK_SHADER_STAGE_ALL_GRAPHICS, RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, sizeof down_factor, &down_factor);
    if (p.uses_render_area) {
        const std::array<float, 4> area{static_cast<float>(targets.extent.width) / up, static_cast<float>(targets.extent.height) / up, 0, 0};
        cmd.PushConstants(*p.layout, VK_SHADER_STAGE_ALL_GRAPHICS, RENDERAREA_LAYOUT_OFFSET, sizeof area, area.data());
    }
    const auto& v = rs.viewports[0];
    const auto& dr = rs.depth_ranges[0];
    if (v[2] <= 0 || v[3] <= 0) throw NotYet("an empty viewport");
    VkViewport vp{v[0] * up, v[1] * up, v[2] * up, v[3] * up, std::clamp(dr[0], 0.0f, 1.0f), std::clamp(dr[1], 0.0f, 1.0f)};
    if (clip.window_origin == 1) vp.y += vp.height, vp.height = -vp.height; // upper left: clip y = +1 is row y
    cmd.SetViewport(0, vp);
    if (rs.color_count && rs.colors[0].texture.address) NoteTargetViewport(rs.colors[0].texture, v[2], v[3]);
    else if (rs.has_depth && rs.depth.texture.address) NoteTargetViewport(rs.depth.texture, v[2], v[3]);
    VkRect2D sc{{0, 0}, targets.extent};
    if (rs.scissor_set[0]) {
        const auto& q = rs.scissors[0];
        const auto in_targets = [&](s32 c) { return static_cast<s32>(static_cast<s64>(c) * targets.scale_n / 4); };
        const s32 x0 = std::max(in_targets(q[0]), 0), y0 = std::max(in_targets(q[1]), 0);
        const s32 x1 = std::min<s32>(in_targets(q[0] + q[2]), targets.extent.width);
        const s32 y1 = std::min<s32>(in_targets(q[1] + q[3]), targets.extent.height);
        sc = {{x0, y0}, {static_cast<u32>(std::max(x1 - x0, 0)), static_cast<u32>(std::max(y1 - y0, 0))}};
    }
    cmd.SetScissor(0, sc);
    cmd.SetStencilReference(VK_STENCIL_FACE_FRONT_BIT, rs.stencil_ref[0]);
    cmd.SetStencilReference(VK_STENCIL_FACE_BACK_BIT, rs.stencil_ref[1]);
    cmd.SetStencilCompareMask(VK_STENCIL_FACE_FRONT_BIT, rs.stencil_value_mask[0]);
    cmd.SetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT, rs.stencil_value_mask[1]);
    cmd.SetStencilWriteMask(VK_STENCIL_FACE_FRONT_BIT, rs.stencil_mask[0]);
    cmd.SetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT, rs.stencil_mask[1]);
    const float blend_constants[4] = {};
    cmd.SetBlendConstants(blend_constants);
    if (rs.depth_bounds_enable && targets.depth != VK_FORMAT_UNDEFINED) cmd.SetDepthBounds(rs.depth_bounds[0], rs.depth_bounds[1]);
    for (const auto& [s, slice] : vbufs) cmd.BindVertexBuffer(s, slice.buffer, slice.offset);
    const u32 instances = static_cast<u32>(std::max(r.instances, 0));
    if (gpu_params) {
        if (r.indexed) {
            cmd.BindIndexBuffer(ibuf.buffer, ibuf.offset, itype);
            cmd.DrawIndexedIndirect(gpu_params->buffer, gpu_params->offset, 1, 20);
        } else {
            cmd.DrawIndirect(gpu_params->buffer, gpu_params->offset, 1, 16);
        }
    } else if (r.indexed) {
        cmd.BindIndexBuffer(ibuf.buffer, ibuf.offset, itype);
        cmd.DrawIndexed(static_cast<u32>(r.count), instances, 0, static_cast<u32>(r.first), static_cast<u32>(r.base_instance));
    } else {
        cmd.Draw(static_cast<u32>(r.count), instances, static_cast<u32>(r.first), static_cast<u32>(r.base_instance));
    }
}

std::atomic<u64> g_dispatched{0}, g_dispatch_skipped{0};

/// `gpu_params`: the group counts' GPU copy (DispatchComputeIndirect with counts only the GPU has); x, y, z are
/// then unknown (0 in c0 until the GPU copies them there).
void Dispatch(Translator& t, u32 x, u32 y, u32 z, const UploadRing::Slice* gpu_params = nullptr) {
    PendingWrites pending(t);
    RenderState& rs = *t.render;
    rs.dispatch = {x, y, z};
    t.after_dispatch.clear(), t.after_dispatch_sizes.clear(); // left by a dispatch that stopped at NotYet
    std::optional<PhaseTimer> phase;
    phase.emplace(kPipeline);
    const ComputePipeline& p = ComputePipelineFor(rs);
    phase.reset();
    std::vector<Vulkan::DescriptorUpdateEntry> entries;
    std::vector<VkDescriptorBufferInfo> c0_copies;
    RescaleBits rescale;
    {
        struct Capture { // t_c0_copies for this scope, also when a descriptor throws NotYet
            explicit Capture(std::vector<VkDescriptorBufferInfo>* v) { t_c0_copies = v; }
            ~Capture() { t_c0_copies = nullptr; }
        } capture(gpu_params ? &c0_copies : nullptr);
        AddDescriptors(t, 5, p.program, p.info, entries, rescale);
    }
    const auto& cmd = t.cmd;
    phase.emplace(kRecord); // the binds, the dispatch, its barriers and copies
    EndOpenPass(cmd);
    if (!c0_copies.empty()) { // the group counts into c0 (kComputeC0Groups)
        for (const auto& c : c0_copies) cmd.CopyBuffer(gpu_params->buffer, c.buffer, VkBufferCopy{gpu_params->offset, c.offset + kComputeC0Groups, 12});
        FullBarrier(cmd);
    }
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *p.pipeline);
    BindDescriptors(t, VK_PIPELINE_BIND_POINT_COMPUTE, p, entries);
    using namespace Shader::Backend::SPIRV;
    const RescaleBits::Words rescaling = RescalingWords(rescale.scaled);
    cmd.PushConstants(*p.layout, VK_SHADER_STAGE_COMPUTE_BIT, RESCALING_LAYOUT_WORDS_OFFSET, sizeof rescaling, rescaling.data());
    if (t.time_dispatch) t.time_dispatch(p.hash, false);
    if (gpu_params) cmd.DispatchIndirect(gpu_params->buffer, gpu_params->offset);
    else cmd.Dispatch(x, y, z);
    if (t.time_dispatch) t.time_dispatch(p.hash, true);
    FullBarrier(cmd);
    if (!t.after_dispatch.empty()) { // the device-memory results into their download slices (ShaderWrite)
        for (size_t i = 0; i < t.after_dispatch.size(); ++i) {
            const auto& [device, slice] = t.after_dispatch[i];
            cmd.CopyBuffer(device.buffer, slice.buffer, VkBufferCopy{device.offset, slice.offset, t.after_dispatch_sizes[i]});
            if (t.bytes) t.bytes->from_device += t.after_dispatch_sizes[i];
        }
        FullBarrier(cmd);
        t.after_dispatch.clear(), t.after_dispatch_sizes.clear();
    }
}

std::atomic<u64> g_draw_log_frame{0}; // SetDrawLogFrame

/// One "nvn drawlog" line for a draw (F9, window.cpp): its stages' content hashes (as the "new pipeline"
/// lines), the draw, the targets, and the depth, stencil, polygon, colour and blend state as bound;
/// `why`: the skip reason, if skipped.
void LogDraw(const RenderState& rs, u64 frame, bool indirect, bool gpu_params, const DrawRecord& r, const std::string& why,
             const std::string& reads) {
    static std::atomic<u64> n{0};
    std::string stages;
    char b[160];
    for (u32 s = 0; s < kStages; ++s) {
        if (!rs.stages[s]) continue;
        std::snprintf(b, sizeof b, " %u:%016llx", s, (unsigned long long)StageContentHash(*rs.stages[s]));
        stages += b;
    }
    std::string targets;
    for (u32 i = 0; i < rs.color_count; ++i) {
        const auto& tex = rs.colors[i].texture;
        if (!tex.address) continue;
        std::snprintf(b, sizeof b, " c%u=0x%llx/0x%x/%ux%u", i, (unsigned long long)tex.address, tex.settings.format,
                      tex.settings.width, tex.settings.height);
        targets += b;
    }
    if (rs.has_depth && rs.depth.texture.address) {
        const auto& tex = rs.depth.texture;
        std::snprintf(b, sizeof b, " z=0x%llx/0x%x", (unsigned long long)tex.address, tex.settings.format);
        targets += b;
    }
    const DepthStencilState ds = std::bit_cast<DepthStencilState>(rs.depth_stencil);
    const PolygonState poly = std::bit_cast<PolygonState>(rs.polygon);
    const ColorState cs = std::bit_cast<ColorState>(rs.color);
    std::string blend;
    for (u32 i = 0; i < rs.color_count; ++i) {
        if (!(cs.blend_enable & (1u << i))) continue;
        const BlendState bl = std::bit_cast<BlendState>(rs.blend[i]);
        std::snprintf(b, sizeof b, " %u:%u/%u,%u/%u,eq%02x", i, bl.src, bl.dst, bl.src_alpha, bl.dst_alpha, bl.equations);
        blend += b;
    }
    std::string layout; // attributes a<i>=format@stream+offset, then streams s<n>=address+size/stride,divisor
    for (u32 i = 0; i < rs.vertex_attrib_count; ++i) {
        const u32 bits = rs.vertex_attribs[i];
        if (!(bits & 0xFF)) continue;
        std::snprintf(b, sizeof b, " a%u=0x%x@%u+%u", i, bits & 0xFF, (bits >> 8) & 0x1F, bits >> 13);
        layout += b;
    }
    for (u32 s = 0; s < rs.vertex_stream_count; ++s) {
        std::snprintf(b, sizeof b, " s%u=0x%llx+0x%llx/%u,%u", s, (unsigned long long)rs.vertex_buffers[s][0],
                      (unsigned long long)rs.vertex_buffers[s][1], rs.vertex_streams[s][0], rs.vertex_streams[s][1]);
        layout += b;
    }
    const auto& dr = rs.depth_ranges[0];
    Log("nvn drawlog f%llu #%llu:%s%s mode %u %s %d x%d (first %d, base instance %d); stages%s; targets%s; vertex%s; depth "
        "test %u write %u func %u range %.3f-%.3f; stencil %u funcs 0x%02x ops 0x%04x/0x%04x ref %u/%u masks 0x%x/0x%x "
        "0x%x/0x%x; cull %u front %u offset 0x%x (%g, %g, %g); channels 0x%08x; blend%s; reads%s%s%s",
        (unsigned long long)frame, (unsigned long long)n.fetch_add(1), indirect ? " indirect" : "",
        gpu_params ? " (GPU parameters)" : "", r.mode, r.indexed ? "indexed" : "arrays", r.count, r.instances, r.first,
        r.base_instance, stages.c_str(), targets.c_str(), layout.c_str(), ds.flags & 1, (ds.flags >> 1) & 1, ds.depth_func, dr[0], dr[1],
        (ds.flags >> 2) & 1, ds.stencil_funcs, ds.front_ops, ds.back_ops, rs.stencil_ref[0], rs.stencil_ref[1],
        rs.stencil_value_mask[0], rs.stencil_value_mask[1], rs.stencil_mask[0], rs.stencil_mask[1], poly.cull_face,
        poly.front_face, poly.offset_enables, rs.polygon_offset[0], rs.polygon_offset[1], rs.polygon_offset[2],
        std::bit_cast<u32>(rs.channel_mask), blend.empty() ? " off" : blend.c_str(), reads.c_str(), why.empty() ? "" : "; SKIPPED: ",
        why.c_str());
}

} // namespace

void SetDrawLogFrame(u64 frame) { g_draw_log_frame.store(frame, std::memory_order_relaxed); }

namespace {
std::atomic<bool> g_skip_toggled{false}; // F10: skip the RT_NVN_TOGGLE_PROGRAMS dispatches

/// Whether the bound compute program is one F10 skips now.
bool SkipToggled(const RenderState& rs) {
    static const std::set<u64> programs = [] {
        std::set<u64> s;
        std::istringstream in(std::getenv("RT_NVN_TOGGLE_PROGRAMS") ? std::getenv("RT_NVN_TOGGLE_PROGRAMS") : "");
        for (std::string x; std::getline(in, x, ',');) {
            if (!x.empty()) s.insert(std::stoull(x, nullptr, 16));
        }
        return s;
    }();
    return g_skip_toggled.load(std::memory_order_relaxed) && rs.compute_stage && !programs.empty() &&
           programs.count(StageContentHash(*rs.compute_stage));
}
} // namespace

bool FlipToggledPrograms() { // the presenting thread only
    const bool on = !g_skip_toggled.load(std::memory_order_relaxed);
    g_skip_toggled.store(on, std::memory_order_relaxed);
    return on;
}

bool TranslateDrawRecord(Translator& t, Op op, const u8* p, size_t size) {
    static const bool skip = std::getenv("RT_NVN_SKIP_DRAWS") != nullptr;
    if (op == Op::kDispatch || op == Op::kDispatchIndirect) {
        u64 v;
        std::memcpy(&v, p, 8);
        u32 x = static_cast<u32>(v), y = static_cast<u32>(v >> 32) & 0xFFFF, z = static_cast<u32>(v >> 48);
        // F9 draw log: the dispatch's program, groups, reads (as draws') and the GPU writes it adds
        const u64 log_frame = g_draw_log_frame.load(std::memory_order_relaxed);
        std::string reads;
        struct ReadLog {
            explicit ReadLog(std::string* s) { t_reads = s; }
            ~ReadLog() { t_reads = nullptr; }
        } read_log(log_frame ? &reads : nullptr);
        // RT_NVN_DUMP_F9=1: the data of an F9-logged dispatch's reads and writes, f9/f<frame>_d<n>_r<k>/w<k>_<address>.bin
        static const bool dump = std::getenv("RT_NVN_DUMP_F9") != nullptr;
        struct DumpPrefix {
            explicit DumpPrefix(std::string s) { t_dump_prefix = std::move(s), t_dump_reads = 0; }
            ~DumpPrefix() { t_dump_prefix.clear(); }
        } dump_prefix([&] {
            static u64 frame = 0;
            static u32 n = 0;
            if (!dump || !log_frame) return std::string();
            if (frame != log_frame) frame = log_frame, n = 0;
            std::filesystem::create_directories("f9");
            return "f9/f" + std::to_string(log_frame) + "_d" + std::to_string(n++) + "_";
        }());
        const size_t writes_before = t.writebacks->size();
        if (SkipToggled(*t.render)) return true; // F10 A/B: as if the program did nothing
        try {
            if (op == Op::kDispatchIndirect) { // v: the counts' address
                u32 g[3];
                std::vector<GpuPart> gpu;
                ReadAsGpuSees(t, v, sizeof g, reinterpret_cast<u8*>(g), &gpu);
                static std::atomic<u64> n{0};
                if (const u64 k = n.fetch_add(1) + 1; k <= 4 || k % 10000 == 0) {
                    Log("nvn dispatch: %llu indirect dispatches (this one at 0x%llx, %s)", (unsigned long long)k,
                        (unsigned long long)v, gpu.empty() ? "counts known at translation" : "GPU-written counts");
                }
                if (!gpu.empty()) {
                    const UploadRing::Slice params = Snapshot(t, v, sizeof g, 4);
                    x = y = z = 0;
                    Dispatch(t, 0, 0, 0, &params);
                } else {
                    x = g[0], y = g[1], z = g[2];
                    Dispatch(t, x, y, z);
                }
            } else if (static const bool test = std::getenv("RT_NVN_INDIRECT_DISPATCH_TEST") != nullptr; test) {
                // test aid: every dispatch through the GPU-counts path (counts in the upload ring, copied into c0
                // on the GPU, vkCmdDispatchIndirect), to check that path with the game's own dispatches
                UploadRing::Slice params = t.upload.Allocate(12, 4);
                const u32 g[3] = {x, y, z};
                std::memcpy(params.data, g, sizeof g);
                Dispatch(t, 0, 0, 0, &params);
            } else {
                Dispatch(t, x, y, z);
            }
            ++g_dispatched;
            if (log_frame) {
                std::string writes;
                for (size_t i = writes_before; i < t.writebacks->size(); ++i) {
                    WriteBack& w = (*t.writebacks)[i];
                    if (!t_dump_prefix.empty()) {
                        char name[48];
                        std::snprintf(name, sizeof name, "w%zu_%llx.bin", i - writes_before, (unsigned long long)w.address);
                        w.dump_name = t_dump_prefix + name;
                    }
                    char b[96];
                    std::snprintf(b, sizeof b, " 0x%llx+0x%llx(%s)", (unsigned long long)w.address, (unsigned long long)w.size,
                                  w.kind == WriteBack::Kind::kRing ? WriterName(w) : "CPU-known");
                    writes += b;
                }
                Log("nvn drawlog f%llu dispatch: program %016llx, %u x %u x %u groups; reads%s; writes%s",
                    (unsigned long long)log_frame, (unsigned long long)StageContentHash(*t.render->compute_stage), x, y, z,
                    reads.c_str(), writes.c_str());
            }
        } catch (const NotYet& e) {
            if (!skip) Fatal("nvn: dispatch: %s is not implemented yet (RT_NVN_SKIP_DRAWS=1 skips such dispatches)", e.what());
            ++g_dispatch_skipped;
            static std::mutex mu;
            static std::set<std::string> reasons;
            std::lock_guard lock(mu);
            if (reasons.insert(e.what()).second) {
                Log("nvn dispatch: skipping dispatches with %s (program 0x%llx)", e.what(),
                    (unsigned long long)t.render->compute_program);
            }
        }
        const u64 n = g_dispatched + g_dispatch_skipped;
        if (n <= 10000 ? n % 1000 == 0 : n % 100000 == 0) {
            Log("nvn dispatch: %llu dispatched, %llu skipped; reads copied on the GPU: %llu (%llu of an earlier "
                "submission's writes)", (unsigned long long)g_dispatched.load(), (unsigned long long)g_dispatch_skipped.load(),
                (unsigned long long)g_gpu_reads.load(), (unsigned long long)g_earlier_reads.load());
        }
        return true;
    }
    if (op != Op::kDraw && op != Op::kDrawIndirect) return false;
    DrawRecord r;
    IndirectDrawRecord ind{};
    if (size < (op == Op::kDraw ? sizeof r : sizeof ind)) Fatal("nvn: a %s record is too short", OpName(op));
    if (op == Op::kDraw) std::memcpy(&r, p, sizeof r);
    else std::memcpy(&ind, p, sizeof ind);
    std::string why;
    UploadRing::Slice params{};
    bool gpu_params = false;
    const u64 log_frame = g_draw_log_frame.load(std::memory_order_relaxed);
    std::string reads;
    struct ReadLog { // t_reads for this draw (F9), also when it throws NotYet
        explicit ReadLog(std::string* s) { t_reads = s; }
        ~ReadLog() { t_reads = nullptr; }
    } read_log(log_frame ? &reads : nullptr);
    try {
        if (op == Op::kDrawIndirect) {
            u32 v[5];
            std::vector<GpuPart> gpu;
            const u64 n = ind.indexed ? 20 : 16;
            ReadAsGpuSees(t, ind.indirect, n, reinterpret_cast<u8*>(v), &gpu);
            gpu_params = !gpu.empty();
            if (gpu_params) {
                // Only the GPU has (part of) the parameters (a compute shader wrote them, often in the
                // submission before; 28k draws in 3 min of battle): drawn from their GPU copy. The index
                // buffer has no size in NVN, so it is copied for the guest memory's values (the last ones the
                // GPU wrote back, frames before) twice over, at least 64 KB, within its memory pool, at most
                // 16 MB; Retire checks the real ones against that (queue.cpp).
                params = Snapshot(t, ind.indirect, n, 4);
                u64 indices = 1;
                if (ind.indexed) {
                    const u64 size = 1ULL << (ind.indexed - 1);
                    u64 bytes = std::clamp<u64>((u64{v[2]} + v[0]) * size * 2, 64 << 10, 16 << 20);
                    if (const PoolRange pool = FindPool(ind.index_address); pool.pool) bytes = std::min(bytes, pool.size - pool.offset);
                    indices = std::max<u64>(bytes / size, 1);
                    if (t.indirect_checks) t.indirect_checks->push_back({ind.indirect, indices});
                }
                static std::atomic<u64> drawn{0};
                if (const u64 k = drawn.fetch_add(1) + 1; k <= 4 || k % 10000 == 0) {
                    Log("nvn draw: %llu indirect draws from GPU-written parameters (this one at 0x%llx, %s, %llu indices copied)",
                        (unsigned long long)k, (unsigned long long)ind.indirect, ind.indexed ? "indexed" : "arrays",
                        (unsigned long long)indices);
                }
                r = {ind.mode, ind.indexed, static_cast<s32>(v[ind.indexed ? 3 : 2]), static_cast<s32>(indices),
                     ind.index_address, static_cast<s32>(v[ind.indexed ? 4 : 3]), 1};
            } else {
                r = ind.indexed ? DrawRecord{ind.mode, ind.indexed, static_cast<s32>(v[3]), static_cast<s32>(v[0]),
                                             ind.index_address + static_cast<u64>(v[2]) * (1ULL << (ind.indexed - 1)),
                                             static_cast<s32>(v[4]), static_cast<s32>(v[1])}
                                : DrawRecord{ind.mode, 0, static_cast<s32>(v[2]), static_cast<s32>(v[0]), 0,
                                             static_cast<s32>(v[3]), static_cast<s32>(v[1])};
            }
        }
        if (gpu_params) Draw(t, r, &params);
        else if (r.count > 0 && r.instances > 0) Draw(t, r);
        ++g_drawn;
        ProfileDrawDone();
    } catch (const NotReady&) { // its pipeline is still being built (pipeline.cpp): drawn once it is
        ++g_waiting;
    } catch (const NotYet& e) {
        why = e.what();
        if (!skip) Fatal("nvn: draw: %s is not implemented yet (RT_NVN_SKIP_DRAWS=1 skips such draws)", e.what());
        ++g_skipped;
    }
    PhaseTimer book(kBook);
    const RenderState& rs = *t.render;
    // the first 50 skipped draws are logged as F9 logs a draw (frame 0 outside F9), to see why without F9
    static std::atomic<u32> skips_logged{0};
    if (log_frame || (!why.empty() && skips_logged.fetch_add(1, std::memory_order_relaxed) < 50)) {
        LogDraw(rs, log_frame, op == Op::kDrawIndirect, gpu_params, r, why, reads);
    }
    static std::mutex mu;
    static std::set<std::tuple<std::array<u64, kStages>, u32, u32, std::array<u32, 9>>> seen;
    static std::map<std::string, u64> reasons; // skip reason ("[indirect]" for indirect draws) -> count
    std::array<u32, 9> formats{};
    for (u32 i = 0; i < rs.color_count; ++i) formats[i] = rs.colors[i].texture.address ? rs.colors[i].texture.settings.format : 0;
    formats[8] = rs.has_depth && rs.depth.texture.address ? rs.depth.texture.settings.format : 0;
    {
        std::lock_guard lock(mu);
        if (!why.empty()) {
            if (op == Op::kDrawIndirect) why += " [indirect]";
            if (reasons[why]++ == 0) Log("nvn draw: skipping draws with %s", why.c_str());
        }
        if (const u64 n = g_drawn + g_skipped; n % 100000 == 0 && !reasons.empty()) {
            std::string list;
            for (const auto& [reason, count] : reasons) list += "; " + std::to_string(count) + " " + reason;
            Log("nvn draw: skips by reason%s", list.c_str());
        }
        std::array<u64, kStages> codes{};
        for (u32 s = 0; s < kStages; ++s) codes[s] = rs.stages[s] ? rs.stages[s]->code : 0;
        if (seen.emplace(codes, r.mode, r.indexed, formats).second && seen.size() <= 200) {
            std::string stages, targets;
            for (u32 s = 0; s < kStages; ++s) {
                if (!rs.stages[s]) continue;
                char b[64];
                std::snprintf(b, sizeof b, " %u@0x%llx", s, (unsigned long long)rs.stages[s]->code);
                stages += b;
            }
            for (u32 i = 0; i < 9; ++i) {
                if (!formats[i]) continue;
                char b[24];
                std::snprintf(b, sizeof b, " %s0x%x", i == 8 ? "z:" : "", formats[i]);
                targets += b;
            }
            Log("nvn draw: #%zu stages%s; mode %u, %s %d, %u attribs, %u streams; targets%s: %s", seen.size(),
                stages.c_str(), r.mode, r.indexed ? "indexed" : "arrays", r.count, rs.vertex_attrib_count,
                rs.vertex_stream_count, targets.c_str(), why.empty() ? "drawn" : "skipped");
        }
    }
    const u64 n = g_drawn + g_skipped;
    if (n <= 10000 ? n % 1000 == 0 : n % 1000000 == 0) { // (every 1000 draws filled 30 % of a player's log)
        Log("nvn draw: %llu drawn, %llu skipped, %llu waited for their pipeline", (unsigned long long)g_drawn.load(),
            (unsigned long long)g_skipped.load(), (unsigned long long)g_waiting.load());
    }
    return true;
}

void TranslateCopyBufferToBuffer(Translator& t, const CopyBufferRecord& r) {
    const auto from = t.upload.Allocate(r.size, 16);
    FillAsGpuSees(t, r.src, r.size, from);
    const auto to = t.download->Allocate(r.size, 16);
    EndOpenPass(t.cmd);
    t.cmd.CopyBuffer(from.buffer, to.buffer, VkBufferCopy{from.offset, to.offset, r.size});
    FullBarrier(t.cmd);
    t.writebacks->push_back({WriteBack::Kind::kRing, r.dst, r.size, to.data, {}, 0, to.buffer, to.offset});
    t.writebacks->back().origin = WriteBack::Origin::kCopyBuffer;
}

} // namespace rt::nvn
