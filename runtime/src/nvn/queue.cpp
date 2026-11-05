// NVNqueue and NVNsync: submission, syncs and write-backs,
// and the queue builder.
//
// QueueSubmitCommands translates the submitted command lists into one Vulkan
// command buffer, on the submitting thread (principle 3), and submits it with
// the queue's next timeline-semaphore value. A submission retires once Vulkan
// has finished it: its write-backs (counter reports so far)
// are written to guest memory first, then the queue's retired value moves
// on. A sync is (queue, value) and is signaled when the queue's retired value
// reaches it, so whoever sees a sync signaled sees the submission's writes.
// Retiring happens whenever the layer looks: at each submission, wait and
// finish. NVN signals a fence recorded in a command list once the commands
// before it are done; the layer signals it at the end of its submission, never
// earlier.
//
// Counters (commands.cpp records them): a report is NVN's 16 bytes {value,
// Switch GPU ticks (614.4 MHz)}, the ticks a Vulkan timestamp at the record's
// place (the game compares its own reports only: it never asks the device's
// clock). TIMESTAMP (type 0) reports value 0; SAMPLES_PASSED (type 1) the
// samples counted since the queue's last ResetCounter of it, as the GPU keeps
// one such counter per queue across submissions. The samples are counted with
// occlusion queries, one per stretch of commands (SampleCounting, images.h),
// from the queue's first type-1 record on, and summed when the submission
// retires. Reports are write-backs (WriteBack::kCounter), applied in command
// order with the others.
//
// RT_TRACE_WAITS (svc.cpp) adds two rows per submission to waits.csv, to see how long a counter report
// takes to reach guest memory (a game may poll it, as Brothership's RenderThread does each frame):
//   submit,<uptime ms at the call>,<ms until vkQueueSubmit returned>,<guest thread>,submit,q<queue>:<value>,,
//          first=<index of its first record in the game's submission (a part after a split: Submit)>;records=<n>;translate=<ms>;ts=<TIMESTAMP reports>@<first>-<last record index>;sp=<SAMPLES_PASSED ...>;
//          sp_cpu=<ms into translation of the last SAMPLES_PASSED report>
//   retire,<uptime ms>,<ms since vkQueueSubmit returned>,,retire,q<queue>:<value>,,
//          gpu=<ms from the submission's first to last command on the GPU>;sp_gpu=<ms of its last SAMPLES_PASSED
//          report after the first command>
//          ;slowest=<the three slowest stretches of kCheckpointRecords records on the GPU: first-last record:ms>
// (traced submissions get a timestamp at their start and end, two queries after the reports', and one after every
// kCheckpointRecords records after those; a submission over RT_GPU_SLOW_MS (default 50) ms on the GPU also logs
// "nvn gpu: slow submission", its slowest stretches and their record types - the new-game run's 2.1 s cutscene
// frames, 2026-10-24; they were compute dispatches all along, so each dispatch now gets a timestamp before and after
// it too, and the report adds the GPU time per compute program, by the code hash the pipeline lines print, and
// the bytes its draws and dispatches copied: TraceBytes, images.h).
#include "heavy.h"
#include "images.h"
#include "render.h"
#include "vk.h"

#include "../kernel.h"
#include "../threads.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <thread>

namespace rt::nvn {

namespace {

// --- QueueBuilder -----------------------------------------------------------------------------------

struct QueueBuilder { // inline in NVNqueueBuilder (0x30 bytes)
    u64 device;
    u32 flags;
    u32 command_size, compute_size, control_size;
    u64 memory, memory_size;
    u32 flush_threshold;
    u16 priority;
    u16 set; // sizes set explicitly (kSet*): the getters report it, as the SDK's
};
constexpr u16 kSetCommand = 1, kSetCompute = 2, kSetControl = 4, kSetFlush = 8;
static_assert(sizeof(QueueBuilder) == 0x30);

} // namespace

// --- QueueBuilder (defaults and getter results as the SDK's: RT_NVN_DIFFTEST) ----------------------


NVN_PROC(nvnQueueBuilderSetDevice, void, u64 builder, u64 device) {
    auto b = Load<QueueBuilder>(builder);
    b.device = device;
    Store(builder, b);
}
// After SetDefaults no size is set: the getters return false and 0 (the driver
// would use its own sizes); the priority is NVN_QUEUE_PRIORITY_NORMAL (100).
NVN_PROC(nvnQueueBuilderSetDefaults, void, u64 builder) {
    auto b = Load<QueueBuilder>(builder);
    const u64 device = b.device;
    b = QueueBuilder{};
    b.device = device;
    b.priority = 100;
    Store(builder, b);
}
#define QB_SET(fn, field, bit)                                                                              \
    NVN_PROC(fn, void, u64 builder, u32 v) {                                                               \
        auto b = Load<QueueBuilder>(builder);                                                              \
        b.field = static_cast<decltype(b.field)>(v);                                                       \
        b.set = static_cast<u16>(b.set | (bit));                                                            \
        Store(builder, b);                                                                                  \
    }
QB_SET(nvnQueueBuilderSetFlags, flags, 0)
QB_SET(nvnQueueBuilderSetCommandMemorySize, command_size, kSetCommand)
QB_SET(nvnQueueBuilderSetComputeMemorySize, compute_size, kSetCompute)
QB_SET(nvnQueueBuilderSetControlMemorySize, control_size, kSetControl)
QB_SET(nvnQueueBuilderSetCommandFlushThreshold, flush_threshold, kSetFlush)
QB_SET(nvnQueueBuilderSetQueuePriority, priority, 0)
#undef QB_SET
NVN_PROC(nvnQueueBuilderSetQueueMemory, void, u64 builder, u64 memory, u64 size) {
    auto b = Load<QueueBuilder>(builder);
    b.memory = memory;
    b.memory_size = size;
    Store(builder, b);
}
NVN_PROC(nvnQueueBuilderGetDevice, u64, u64 builder) { return Load<QueueBuilder>(builder).device; }
NVN_PROC(nvnQueueBuilderGetFlags, u32, u64 builder) { return Load<QueueBuilder>(builder).flags; }
#define QB_GET(fn, field, bit)                                                                              \
    NVN_PROC(fn, bool, u64 builder, u64 out) {                                                             \
        const auto b = Load<QueueBuilder>(builder);                                                        \
        hle::WriteGuest<s32>(out, static_cast<s32>(b.field));                                              \
        return (b.set & (bit)) != 0;                                                                        \
    }
QB_GET(nvnQueueBuilderGetCommandMemorySize, command_size, kSetCommand)
QB_GET(nvnQueueBuilderGetComputeMemorySize, compute_size, kSetCompute)
QB_GET(nvnQueueBuilderGetControlMemorySize, control_size, kSetControl)
QB_GET(nvnQueueBuilderGetCommandFlushThreshold, flush_threshold, kSetFlush)
#undef QB_GET
NVN_PROC(nvnQueueBuilderGetQueuePriority, u32, u64 builder) { return Load<QueueBuilder>(builder).priority; }
NVN_PROC(nvnQueueBuilderGetMemory, u64, u64 builder) { return Load<QueueBuilder>(builder).memory; }
// GetQueueMemorySize / GetMemorySize: the SDK computes them from the device
// (it reads the builder's device); not implemented until a game needs them.

namespace {

// --- Queue ---------------------------------------------------------------------------------------

struct Queue;

struct Sync : Object {
    static constexpr Type kType = Type::kSync;
    bool host_signaled = false; // signaled by the layer itself (a window's acquire), until fenced again
    Queue* queue = nullptr; // fenced on (nullptr: never fenced)
    u64 value = 0;          // signaled once queue->retired >= value
};
std::mutex g_sync_mu; // Sync::queue and Sync::value

/// A GPU wait on memory that holds no initialised sync: NVN reads the fence values left there — of a
/// finalised sync, long reached; of zeroed memory, syncpoint 0 value 0, reached — and does not wait.
void WaitOnStaleSync(u64 sync) {
    static std::once_flag once;
    std::call_once(once, [&] { Log("nvn: a wait on 0x%llx, which holds no initialised sync: no wait (as NVN; logged once)", (unsigned long long)sync); });
}
std::mutex g_queues_mu;
std::set<struct Queue*> g_queues; // every initialised queue, for RetireAll

struct CounterEvent { // a ResetCounter or ReportCounter, in command order
    bool report;
    u32 counter;
    u32 timestamp; // report: its timestamp query
    u32 segments;  // occlusion queries ended before it
};
constexpr u32 kSegmentsPerPool = 256;
constexpr u32 kCheckpointRecords = 32; // RT_TRACE_WAITS: records between GPU timestamps

struct EventRecord { // Op::kSignalEvent; also an event write a submission makes when it retires
    u64 address;     // the event's word (its pool storage + offset)
    u32 mode, value; // mode 0: write, 1: add
};

struct FillRecord { // Op::kClearBuffer: a GPU fill of guest memory, applied when the submission retires
    u64 address, size;
    u32 value, pad;
};

void ApplyEvent(const EventRecord& e) {
    const u32 old = hle::ReadGuest<u32>(e.address);
    hle::WriteGuest<u32>(e.address, e.mode == 1 ? old + e.value : e.value);
}

void Apply(const WriteBack& w) {
    switch (w.kind) {
    case WriteBack::Kind::kRing: g_mem.Write(w.address, w.ring, w.size); break;
    case WriteBack::Kind::kBytes: g_mem.Write(w.address, w.bytes.data(), w.bytes.size()); break;
    case WriteBack::Kind::kFill: {
        const std::vector<u32> words(w.size / 4, w.value);
        g_mem.Write(w.address, words.data(), words.size() * 4);
        break;
    }
    case WriteBack::Kind::kEventWrite: ApplyEvent({w.address, 0, w.value}); break;
    case WriteBack::Kind::kEventAdd: ApplyEvent({w.address, 1, w.value}); break;
    case WriteBack::Kind::kCounter: break; // Retire writes it
    }
}

struct Submission {
    u64 value = 0;
    u64 submitted_ns = 0; // RT_TRACE_PRESENT: when vkQueueSubmit returned
    Vulkan::vk::CommandBuffers buffers; // one
    Vulkan::vk::QueryPool queries;
    u32 query_capacity = 0;
    std::vector<CounterEvent> counters;
    std::vector<Vulkan::vk::QueryPool> segment_pools; // occlusion queries, kSegmentsPerPool each
    u32 segments = 0;                                  // used in this submission
    std::vector<WriteBack> writes;             // GPU writes to guest memory, applied in order at retirement
    UploadRing download{Vulkan::MemoryUsage::Download}; // GPU results read back (CopyTextureToBuffer ...)
    UploadRing upload;                         // guest memory read at submit (5.0 rule 1)
    DescriptorArena sets;                      // for pipelines over the push-descriptor limit
    std::vector<IndirectCheck> indirect_checks; // indexed draws with GPU-written parameters
    UploadRing device{Vulkan::MemoryUsage::DeviceLocal}; // compute shaders' writes, before their download copy
    std::vector<std::shared_ptr<const void>> keep; // images and the like the commands use
    u64 keep_until = 0; // a later submission copies from this one's download ring: not reused before it retires
    bool traced = false;      // RT_TRACE_WAITS: start and end timestamps after the reports' queries
    s32 last_sample = -1;     // RT_TRACE_WAITS: query of the last SAMPLES_PASSED report, -1: none
    u32 checkpoints = 0;      // RT_TRACE_WAITS: timestamps after every kCheckpointRecords records (slots planned)
    u32 checkpoints_written = 0; // of them at a stretch's end (the rest filled at the part's end: a cut, CutMs)
    u32 report_slots = 0;     // queries planned for counter reports: the traced timestamps follow them
    size_t first_record = 0;  // RT_TRACE_WAITS: the part's first record in the game's submission
    std::vector<Op> ops;      // RT_TRACE_WAITS: the part's record types, for a slow submission's report
    u32 dispatch_slots = 0;   // RT_TRACE_WAITS: two timestamps per dispatch record, after the checkpoints
    std::vector<u64> dispatch_hashes; // RT_TRACE_WAITS: the timed dispatches' compute code, in slot order
    TraceBytes bytes;                 // RT_TRACE_WAITS: what its draws and dispatches copied
    std::vector<CacheVerify> cache_verify; // RT_NVN_BUFFER_CACHE_CHECK: reads to compare at retirement
    HeavyProfile heavy;                    // RT_PROFILE_HEAVY (heavy.h)
    WriteIndex index;                      // of `writes` (images.h; draws.cpp IndexedRead)
};

struct Queue : Object {
    static constexpr Type kType = Type::kQueue;
    u64 device = 0;
    QueueBuilder settings{};
    std::mutex mu; // everything below
    Vulkan::vk::Semaphore timeline;
    Vulkan::vk::CommandPool pool;
    std::deque<std::unique_ptr<Submission>> in_flight;
    std::vector<std::unique_ptr<Submission>> spare;
    u64 submitted = 0;
    std::atomic<u64> retired{0};
    std::vector<std::pair<Queue*, u64>> waits; // other queues' syncs the next submission waits for
    RenderState render;                        // NVN's render state on this queue (render.h)
    bool count_samples = false;                // a SAMPLES_PASSED record was translated: count from then on
    u64 samples = 0;                           // the GPU's SAMPLES_PASSED counter, at the last retirement
    u32 trace_id = 0;                          // RT_TRACE_WAITS: q<n> in waits.csv
    // Retirement on its own thread, as the GPU writes results whether or not the game calls NVN (a game may
    // poll a counter report in memory): wakes on submissions, waits on the timeline, retires.
    std::condition_variable work;
    bool stopping = false;
    std::thread retirer;
    std::shared_ptr<BufferCache> cache; // the buffer cache (draws.cpp), made at the first submission
    bool cache_made = false;
};

struct Vulkan_ {
    PFN_vkCmdWriteTimestamp write_timestamp = nullptr;
    double ns_per_tick = 0; // of Vulkan timestamps
    u64 valid_mask = 0;     // of timestamp bits
};
const Vulkan_& Vkq() {
    static const Vulkan_ v = [] {
        Vulkan_ r;
        const auto& d = *Vk().device;
        r.write_timestamp = reinterpret_cast<PFN_vkCmdWriteTimestamp>(
            Vk().dld.vkGetDeviceProcAddr(*d.GetLogical(), "vkCmdWriteTimestamp"));
        const auto props = d.GetPhysical().GetProperties();
        const auto families = d.GetPhysical().GetQueueFamilyProperties();
        const u32 bits = families.at(d.GetGraphicsFamily()).timestampValidBits;
        if (!r.write_timestamp || !bits) Fatal("nvn: the Vulkan device has no timestamps on its graphics queue");
        if (!d.HasTimelineSemaphore()) Fatal("nvn: the Vulkan device has no timeline semaphores");
        r.ns_per_tick = props.limits.timestampPeriod;
        r.valid_mask = bits >= 64 ? ~0ULL : (1ULL << bits) - 1;
        return r;
    }();
    return v;
}

/// Switch GPU ticks (614.4 MHz) of a Vulkan timestamp.
u64 SwitchTicks(u64 timestamp) {
    const double ns = static_cast<double>(timestamp & Vkq().valid_mask) * Vkq().ns_per_tick;
    return static_cast<u64>(ns * 384.0 / 625.0);
}

std::atomic<u64> g_submit_ns{0}, g_wait_ns{0}, g_submits{0}, g_gpu_max_ns{0};

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

bool TraceSubmits() {
    static const bool trace = std::getenv("RT_TRACE_PRESENT") != nullptr;
    return trace;
}

/// RT_TRACE_WAITS: the GPU time of each stretch of kCheckpointRecords records of a retired traced submission,
/// as (ms, stretch); `ts` = its start, end and checkpoint timestamps.
std::vector<std::pair<double, u32>> StretchTimes(const Submission& s, const u64* ts) {
    std::vector<u64> at{ts[0]};
    for (u32 i = 0; i < s.checkpoints_written; ++i) at.push_back(ts[2 + i]);
    at.push_back(ts[1]);
    std::vector<std::pair<double, u32>> out;
    for (u32 i = 0; i + 1 < at.size(); ++i) {
        const u64 a = at[i] & Vkq().valid_mask, b = at[i + 1] & Vkq().valid_mask;
        out.push_back({b > a ? static_cast<double>(b - a) * Vkq().ns_per_tick / 1e6 : 0.0, i});
    }
    std::sort(out.rbegin(), out.rend());
    return out;
}

/// "<n> <op>, ..." of records [first, last] of a traced submission's part, most frequent first.
std::string OpSummary(const Submission& s, size_t first, size_t last) {
    std::map<Op, u32> n;
    for (size_t i = first; i <= last && i < s.ops.size(); ++i) ++n[s.ops[i]];
    std::vector<std::pair<u32, Op>> by;
    for (const auto& [op, k] : n) by.push_back({k, op});
    std::sort(by.rbegin(), by.rend());
    std::string out;
    for (const auto& [k, op] : by) out += (out.empty() ? "" : ", ") + std::to_string(k) + " " + OpName(op);
    return out;
}

/// RT_PROFILE_HEAVY: a record's kind for heavy.h (draw 0, dispatch 1, other 2).
u8 HeavyKind(Op op) {
    if (op == Op::kDraw || op == Op::kDrawIndirect) return 0;
    if (op == Op::kDispatch || op == Op::kDispatchIndirect) return 1;
    return 2;
}

/// RT_PROFILE_HEAVY: a ring write-back's bytes against guest memory just before it is applied (which of the bytes
/// the GPU left as it found them were changed by the shader, as far as guest memory still holds the old ones).
HeavyWrite CompareWithGuest(const WriteBack& w) {
    thread_local std::vector<u8> now;
    now.resize(w.size);
    g_mem.Read(w.address, now.data(), w.size);
    HeavyWrite h{w.size, 0, 0, 0};
    for (u64 i = 0; i < w.size; ++i) {
        if (now[i] == w.ring[i]) continue;
        if (!h.changed++) h.first = i;
        h.last = i;
    }
    return h;
}

/// Retires the queue's finished submissions: write-backs, then the retired value. With q->mu.
void Retire(Queue* q) {
    const u64 done = q->timeline.GetCounter();
    while (!q->in_flight.empty() && q->in_flight.front()->value <= done) {
        std::unique_ptr<Submission> s = std::move(q->in_flight.front());
        q->in_flight.pop_front();
        const auto& dev = Vk().device->GetLogical();
        u32 timestamps = 0;
        for (const CounterEvent& e : s->counters) timestamps += e.report;
        if (s->traced) timestamps = s->report_slots; // the layout as planned (a part cut early filled the rest)
        const u32 read = timestamps + (s->traced ? 2 + s->checkpoints + s->dispatch_slots : 0);
        std::vector<u64> ts(read), samples(s->segments);
        if (read) {
            const VkResult r = dev.GetQueryResults(*s->queries, 0, read, ts.size() * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT);
            if (r != VK_SUCCESS) Fatal("nvn: timestamp results of a finished submission: VkResult %d", r);
        }
        for (u32 first = 0; first < s->segments; first += kSegmentsPerPool) {
            const u32 n = std::min(kSegmentsPerPool, s->segments - first);
            const VkResult r = dev.GetQueryResults(*s->segment_pools[first / kSegmentsPerPool], 0, n, n * 8,
                                                   samples.data() + first, 8, VK_QUERY_RESULT_64_BIT);
            if (r != VK_SUCCESS) Fatal("nvn: occlusion results of a finished submission: VkResult %d", r);
        }
        std::vector<std::array<u64, 2>> reports; // by WriteBack::kCounter's index
        u32 summed = 0;
        for (const CounterEvent& e : s->counters) {
            for (; summed < e.segments; ++summed) q->samples += samples[summed];
            if (!e.report) {
                if (e.counter == 1) q->samples = 0;
                continue;
            }
            reports.push_back({e.counter == 1 ? q->samples : 0, SwitchTicks(ts[e.timestamp])});
            if (e.counter == 1) {
                static std::atomic<u64> n{0}, nonzero{0};
                const u64 k = n.fetch_add(1) + 1, z = nonzero.fetch_add(q->samples != 0) + (q->samples != 0);
                if (k == 1 || k == 10 || k == 100 || k == 1000 || k % 100000 == 0) {
                    Log("nvn: %llu SAMPLES_PASSED reports, %llu of them non-zero (this one %llu)", (unsigned long long)k,
                        (unsigned long long)z, (unsigned long long)q->samples);
                }
            }
        }
        for (; summed < s->segments; ++summed) q->samples += samples[summed];
        s->counters.clear();
        s->segments = 0;
        s->download.Invalidate(); // 5.0 rules 2-3: GPU writes reach guest memory before the syncs report signaled
        std::vector<HeavyWrite> heavy_writes; // RT_PROFILE_HEAVY: a heavy submission's write-backs, compared
        if (s->heavy.on && s->traced) {
            const u64 a = ts[timestamps] & Vkq().valid_mask, b = ts[timestamps + 1] & Vkq().valid_mask;
            const double gpu = b > a ? static_cast<double>(b - a) * Vkq().ns_per_tick / 1e6 : 0.0;
            if (s->heavy.translate_ns / 1e6 >= HeavyMs() || gpu >= HeavyMs()) heavy_writes.resize(s->writes.size());
        }
        for (const CacheVerify& v : s->cache_verify) { // RT_NVN_BUFFER_CACHE_CHECK: what the GPU found in the cache
            static std::atomic<u64> compared{0}, differ{0};
            const u64 n = compared.fetch_add(1) + 1;
            const u8* want = v.ref ? v.ref : v.expected.data();
            if (std::memcmp(v.got, want, v.expected.size()) != 0) {
                size_t k = 0;
                while (v.got[k] == want[k]) ++k;
                size_t bad = 0;
                for (size_t i = 0; i < v.expected.size(); ++i) bad += v.got[i] != want[i];
                if (v.ref) {
                    static std::atomic<u64> told{0};
                    if (told.fetch_add(1) < 40) {
                        Log("nvn: PAGE COPY CHECK: the page copy of 0x%llx+0x%zx (phase %u, %s, submission %llu) differs from a "
                            "copy of its own: %zu bytes, first at +0x%zx (page copy %02x, own copy %02x)",
                            (unsigned long long)v.address, v.expected.size(), v.phase, v.gpu_parts ? "with GPU parts" : "CPU only",
                            (unsigned long long)s->value, bad, k, v.got[k], want[k]);
                        Log("nvn:   page copy: %s", v.note.c_str());
                    }
                    continue;
                }
                const u64 d = differ.fetch_add(1) + 1;
                if (d <= 40 || d % 1000 == 0) {
                    Log("nvn: buffer cache GPU check: the GPU read of 0x%llx+0x%zx (phase %u, submission %llu) found other bytes in "
                        "the cache (#%llu): %zu bytes differ, first at +0x%zx (cache %02x, expected %02x)",
                        (unsigned long long)v.address, v.expected.size(), v.phase, (unsigned long long)s->value,
                        (unsigned long long)d, bad, k, v.got[k], v.expected[k]);
                }
            }
            static std::atomic<u64> largest{0};
            for (u64 l = largest.load(); v.expected.size() > l && !largest.compare_exchange_weak(l, v.expected.size());) {}
            if (n % 20000 == 0) {
                Log("nvn: buffer cache GPU check: %llu reads compared, %llu differ (largest %.1f MB)", (unsigned long long)n,
                    (unsigned long long)differ.load(), largest.load() / 1048576.0);
            }
        }
        s->cache_verify.clear();
        // RT_NVN_DUMP_WRITES=<size>: the first 200 GPU writes of exactly that many bytes, as the GPU left them,
        // to write_<n>_<address>.bin (a diagnostic: a shader's output read offline).
        static const u64 dump_size = std::getenv("RT_NVN_DUMP_WRITES") ? std::strtoull(std::getenv("RT_NVN_DUMP_WRITES"), nullptr, 0) : 0;
        static u32 dumped = 0;
        const u64 t_writeback = NowNs();
        u64 writeback_bytes = 0; // RT_TRACE_WAITS: the ring write-backs into guest memory, for the report
        for (const WriteBack& w : s->writes) {
            if (w.kind == WriteBack::Kind::kRing) writeback_bytes += w.size;
            if (!w.dump_name.empty() && w.kind == WriteBack::Kind::kRing) { // RT_NVN_DUMP_F9
                if (FILE* f = std::fopen(w.dump_name.c_str(), "wb")) {
                    std::fwrite(w.ring, 1, w.size, f);
                    std::fclose(f);
                }
            }
            if (dump_size && w.kind == WriteBack::Kind::kRing && w.size == dump_size && dumped < 200) {
                char name[64];
                std::snprintf(name, sizeof name, "write_%03u_%llx.bin", dumped++, (unsigned long long)w.address);
                if (FILE* f = std::fopen(name, "wb")) {
                    std::fwrite(w.ring, 1, w.size, f);
                    std::fclose(f);
                }
            }
            if (!heavy_writes.empty() && w.kind == WriteBack::Kind::kRing) heavy_writes[&w - s->writes.data()] = CompareWithGuest(w);
            if (w.kind == WriteBack::Kind::kCounter) hle::WriteGuest(w.address, reports.at(w.value));
            else Apply(w);
        }
        s->writes.clear();
        s->index.Reset();
        const double writeback_ms = (NowNs() - t_writeback) / 1e6;
        // The GPU-written parameters are in guest memory now (their writers retired, in order): did a draw use
        // more indices than draws.cpp copied for it? (Later writes of this submission can make this a guess.)
        for (const IndirectCheck& c : s->indirect_checks) {
            u32 v[5];
            g_mem.Read(c.params, v, sizeof v);
            const u64 need = u64{v[2]} + v[0];
            if (v[0] && v[1] && need > c.indices) {
                static std::atomic<u64> n{0};
                const u64 k = n.fetch_add(1) + 1;
                if (k <= 10 || k % 1000 == 0) {
                    Log("nvn: indirect draw #%llu read past its index copy: parameters at 0x%llx want indices to %llu, %llu copied",
                        (unsigned long long)k, (unsigned long long)c.params, (unsigned long long)need, (unsigned long long)c.indices);
                }
            }
        }
        s->indirect_checks.clear();
        if (s->traced) {
            const auto ms = [&](u64 a, u64 b) { return static_cast<double>((b - a) & Vkq().valid_mask) * Vkq().ns_per_tick / 1e6; };
            const u64 start = ts[timestamps], end = ts[timestamps + 1];
            const double gpu = ms(start, end);
            const std::vector<std::pair<double, u32>> stretches = StretchTimes(*s, &ts[timestamps]);
            const auto span = [&](u32 i) {
                const size_t a = size_t{i} * kCheckpointRecords;
                return std::pair{a, std::min(a + kCheckpointRecords, s->ops.size()) - 1}; // in the part
            };
            std::string slowest;
            for (size_t i = 0; i < stretches.size() && i < 3; ++i) {
                const auto [a, z] = span(stretches[i].second);
                char e[64];
                std::snprintf(e, sizeof e, "%s%zu-%zu:%.3f", i ? "," : "", s->first_record + a, s->first_record + z, stretches[i].first);
                slowest += e;
            }
            char b[384];
            std::snprintf(b, sizeof b, "retire,%.3f,%.3f,,retire,q%u:%llu,,gpu=%.3f;sp_gpu=%.3f;slowest=%s", Uptime() * 1e3,
                          (NowNs() - s->submitted_ns) / 1e6, q->trace_id, (unsigned long long)s->value, gpu,
                          s->last_sample < 0 ? -1.0 : ms(start, ts[s->last_sample]), slowest.c_str());
            TraceWaitsLine(b);
            static const double slow_ms = std::getenv("RT_GPU_SLOW_MS") ? std::atof(std::getenv("RT_GPU_SLOW_MS")) : 50.0;
            if (gpu > slow_ms) {
                Log("nvn gpu: slow submission q%u:%llu (t=%.2f s): %.1f ms on the GPU, records %zu-%zu; slowest stretches:",
                    q->trace_id, (unsigned long long)s->value, Uptime(), gpu, s->first_record, s->first_record + s->ops.size() - 1);
                for (size_t i = 0; i < stretches.size() && i < 3; ++i) {
                    const auto [a, z] = span(stretches[i].second);
                    Log("nvn gpu:   records %zu-%zu: %.1f ms (%s)", s->first_record + a, s->first_record + z,
                        stretches[i].first, OpSummary(*s, a, z).c_str());
                }
                std::map<u64, std::pair<double, u32>> by_code; // compute code: GPU ms, dispatches
                const u64* d = &ts[timestamps + 2 + s->checkpoints];
                for (size_t i = 0; i < s->dispatch_hashes.size(); ++i) {
                    auto& e = by_code[s->dispatch_hashes[i]];
                    e.first += ms(d[2 * i], d[2 * i + 1]), ++e.second;
                }
                std::vector<std::pair<double, u64>> top;
                double all = 0;
                for (const auto& [code, e] : by_code) top.push_back({e.first, code}), all += e.first;
                std::sort(top.rbegin(), top.rend());
                const TraceBytes& y = s->bytes;
                const auto mb = [](u64 n) { return n / 1048576.0; };
                Log("nvn gpu:   copies: %u shader-written buffers filled %.1f MB (largest %.2f MB), reads filled %.1f MB; "
                    "%.1f MB of those fills copied on the GPU from earlier results; compute writes %.1f MB into device "
                    "memory, %.1f MB back; at retirement %.1f MB written into guest memory in %.1f ms (all writes)",
                    y.writes, mb(y.write_fills), mb(y.largest_write), mb(y.read_fills), mb(y.gpu_parts), mb(y.to_device),
                    mb(y.from_device), mb(writeback_bytes), writeback_ms);
                Log("nvn gpu:   %zu dispatches, %.1f ms in them; by compute program:", s->dispatch_hashes.size(), all);
                for (size_t i = 0; i < top.size() && i < 8; ++i) {
                    const auto& e = by_code[top[i].second];
                    Log("nvn gpu:     %016llx: %u dispatches, %.1f ms (%.2f ms each)", (unsigned long long)top[i].second,
                        e.second, e.first, e.first / e.second);
                }
            }
            if (s->heavy.on) { // RT_PROFILE_HEAVY: the stretches' GPU times in record order, each dispatch's
                std::vector<double> stretch_ms, dispatch_ms;
                std::vector<u64> at{ts[timestamps]};
                for (u32 i = 0; i < s->checkpoints_written; ++i) at.push_back(ts[timestamps + 2 + i]);
                at.push_back(ts[timestamps + 1]);
                for (size_t i = 0; i + 1 < at.size(); ++i) stretch_ms.push_back(ms(at[i], at[i + 1]));
                const u64* d = &ts[timestamps + 2 + s->checkpoints];
                for (size_t i = 0; i < s->dispatch_hashes.size(); ++i) dispatch_ms.push_back(ms(d[2 * i], d[2 * i + 1]));
                std::vector<u8> kinds(s->ops.size());
                for (size_t i = 0; i < s->ops.size(); ++i) kinds[i] = HeavyKind(s->ops[i]);
                HeavyRetired(q->trace_id, s->value, s->first_record, gpu, s->heavy, stretch_ms, kCheckpointRecords, dispatch_ms,
                             kinds, heavy_writes);
                s->heavy.on = false;
            }
            s->traced = false;
        }
        if (s->submitted_ns) { // RT_TRACE_PRESENT
            const u64 took = NowNs() - s->submitted_ns;
            u64 seen = g_gpu_max_ns.load(std::memory_order_relaxed);
            while (took > seen && !g_gpu_max_ns.compare_exchange_weak(seen, took)) {}
        }
        s->download.Reset();
        s->upload.Reset();
        s->sets.Reset();
        s->device.Reset();
        s->keep.clear();
        q->retired.store(s->value, std::memory_order_release);
        q->spare.push_back(std::move(s));
    }
}

/// Waits (with the kernel lock released) until the queue has retired `value` or `timeout_ns` passed.
bool WaitRetired(Queue* q, u64 value, u64 timeout_ns) {
    if (q->retired.load(std::memory_order_acquire) >= value) return true;
    const u64 t0 = NowNs();
    RunHostWait(kHostWaitGpu, [&] { q->timeline.Wait(value, timeout_ns); });
    g_wait_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    std::lock_guard lock(q->mu);
    Retire(q);
    return q->retired.load(std::memory_order_acquire) >= value;
}

struct Rec {
    Op op;
    std::vector<u8> payload;
};

template <typename T>
T Payload(const Rec& r) {
    if (r.payload.size() < sizeof(T)) Fatal("nvn: a %s record is too short", OpName(r.op));
    T v;
    std::memcpy(&v, r.payload.data(), sizeof v);
    return v;
}

/// SAMPLES_PASSED on a submission's command buffer: an occlusion query per stretch (images.h), from the
/// submission's pools (reset by the host when taken: the submission has retired or is new).
struct Segments final : SampleCounting {
    const Vulkan::vk::CommandBuffer& cmd;
    Submission& s;
    bool open = false;
    Segments(const Vulkan::vk::CommandBuffer& c, Submission& sub) : cmd(c), s(sub) {}
    void Pause() override {
        if (!open) return;
        cmd.EndQuery(*s.segment_pools[(s.segments - 1) / kSegmentsPerPool], (s.segments - 1) % kSegmentsPerPool);
        open = false;
    }
    void Resume() override {
        const u32 pool = s.segments / kSegmentsPerPool, query = s.segments % kSegmentsPerPool;
        const auto& dev = Vk().device->GetLogical();
        if (pool == s.segment_pools.size()) {
            s.segment_pools.push_back(dev.CreateQueryPool({
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queryType = VK_QUERY_TYPE_OCCLUSION,
                .queryCount = kSegmentsPerPool,
                .pipelineStatistics = 0,
            }));
        }
        if (query == 0) dev.ResetQueryPool(*s.segment_pools[pool], 0, kSegmentsPerPool);
        cmd.BeginQuery(*s.segment_pools[pool], query, VK_QUERY_CONTROL_PRECISE_BIT);
        ++s.segments;
        open = true;
    }
};

/// RT_NVN_SPLIT=<n>: cut a submission after its last counter report that at least n records follow (see
/// Submit; default 256); 0: never, each submission as one Vulkan submission as before 2026-10-19 (A/B).
size_t SplitMinTail() {
    static const size_t n = [] {
        const char* e = std::getenv("RT_NVN_SPLIT");
        return e && *e ? static_cast<size_t>(std::strtoull(e, nullptr, 10)) : size_t{256};
    }();
    return n;
}

/// Records [begin, end) of a submission as one Vulkan submission (one retirement). With q->mu.
/// `waits_in`: other queues' syncs it waits for (a later part waits for them again: harmless, they are
/// reached); `t0`, `t_locked`: when the call began and took q->mu (RT_TRACE_PRESENT, RT_TRACE_WAITS).
/// RT_TRACE_WAITS: the translation time of a traced submission by record type, summed over each second of uptime and
/// logged once it is over ("nvn translate: ..."), to see what a heavy scene's translation is made of
/// (the per-op timer costs ~20 ns a record).
void TranslateSecond(const std::map<Op, std::pair<u64, u32>>& op_ns, u64 translate_ns, size_t records) {
    static std::mutex mu;
    static std::map<Op, std::pair<u64, u64>> sum;
    static u64 second = 0, total_ns = 0, total_records = 0, submissions = 0;
    std::lock_guard lock(mu);
    const u64 now = static_cast<u64>(Uptime());
    if (now != second) {
        if (submissions) {
            std::vector<std::pair<u64, Op>> top;
            for (const auto& [op, e] : sum) top.push_back({e.first, op});
            std::sort(top.rbegin(), top.rend());
            std::string by;
            for (size_t i = 0; i < top.size() && i < 8; ++i) {
                const auto& e = sum[top[i].second];
                char b[96];
                std::snprintf(b, sizeof b, "%s%s %.0f ms (%llu, %.2f us each)", i ? ", " : "", OpName(top[i].second),
                              e.first / 1e6, (unsigned long long)e.second, e.first / 1e3 / std::max<u64>(e.second, 1));
                by += b;
            }
            Log("nvn translate: t=%llu s: %.0f ms in %llu submissions, %llu records: %s", (unsigned long long)second,
                total_ns / 1e6, (unsigned long long)submissions, (unsigned long long)total_records, by.c_str());
        }
        sum.clear(), second = now, total_ns = total_records = submissions = 0;
    }
    for (const auto& [op, e] : op_ns) sum[op].first += e.first, sum[op].second += e.second;
    total_ns += translate_ns, total_records += records, ++submissions;
}

/// RT_NVN_CUT_MS=<ms> (default 4; 0: off, A/B). A part whose translation has taken
/// that long ends at the next record after which no render pass is open, is submitted, and the submission goes on
/// in a new part: the GPU starts on it while the rest is translated (the cutscenes' ~975-dispatch frames: the GPU
/// waited ~100 ms for the whole translation, then needed ~70 ms). Light submissions (~1 ms) never reach it.
u64 CutNs() {
    static const u64 ns = [] {
        const char* e = std::getenv("RT_NVN_CUT_MS");
        const double ms = e && *e ? std::atof(e) : 4.0;
        Log("nvn: long submissions cut every %.1f ms of translation (RT_NVN_CUT_MS; 0: off)", ms);
        return static_cast<u64>(std::max(ms, 0.0) * 1e6);
    }();
    return ns;
}

/// Translates and submits records [begin, end) of a submission, or the first of them up to a cut (CutNs); returns
/// where it stopped. `waits`: other queues' syncs to wait for, in and out (a later part waits for this part's
/// WaitSync records too: its commands come after them).
size_t SubmitPart(Queue* q, const std::vector<Rec>& recs, size_t begin, size_t end,
                  std::vector<std::pair<Queue*, u64>>& waits_io, u64 t0, u64 t_locked) {
    Retire(q);
    std::unique_ptr<Submission> s;
    const u64 retired = q->retired.load(std::memory_order_acquire);
    const auto reusable = std::find_if(q->spare.rbegin(), q->spare.rend(), [&](const auto& e) { return e->keep_until <= retired; });
    if (reusable == q->spare.rend()) {
        s = std::make_unique<Submission>();
        s->buffers = q->pool.Allocate(2); // [0] the commands, [1] the buffer cache's copies before them
    } else {
        s = std::move(*reusable);
        q->spare.erase(std::next(reusable).base());
    }
    s->keep_until = 0;
    s->value = q->submitted + 1;
    const std::span<const Rec> part(recs.data() + begin, end - begin);
    u32 counters = 0;
    for (const Rec& r : part) counters += r.op == Op::kReportCounter;
    s->traced = TraceWaitsOn();
    s->last_sample = -1;
    s->heavy.Reset(s->traced, part.size());
    struct DriverScope { // RT_PROFILE_HEAVY: this thread's Vulkan calls count to the submission until it returns
        explicit DriverScope(DriverTimes* d) { t_driver = d; }
        ~DriverScope() { t_driver = nullptr, t_heavy_phases = nullptr; }
    } driver_scope(s->heavy.on ? &s->heavy.driver : nullptr);
    if (s->traced && !q->trace_id) {
        static std::atomic<u32> ids{0};
        q->trace_id = ids.fetch_add(1) + 1;
    }
    s->checkpoints = s->traced && !part.empty() ? static_cast<u32>((part.size() - 1) / kCheckpointRecords) : 0;
    s->checkpoints_written = 0;
    s->report_slots = counters;
    s->first_record = begin;
    s->ops.clear();
    if (s->traced) for (const Rec& r : part) s->ops.push_back(r.op);
    u32 dispatches = 0;
    if (s->traced) for (const Rec& r : part) dispatches += r.op == Op::kDispatch || r.op == Op::kDispatchIndirect;
    s->dispatch_slots = 2 * dispatches;
    s->dispatch_hashes.clear();
    const u32 dispatch_base = counters + 2 + s->checkpoints;
    const u32 slots = counters + (s->traced ? 2 + s->checkpoints + s->dispatch_slots : 0);
    if (slots > s->query_capacity) {
        s->query_capacity = std::max<u32>(slots, 2 * s->query_capacity);
        s->queries = Vk().device->GetLogical().CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = s->query_capacity,
            .pipelineStatistics = 0,
        });
    }
    const Vulkan::vk::CommandBuffer cmd(s->buffers[0], Vk().device->GetDispatchLoader());
    cmd.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    if (slots) cmd.ResetQueryPool(*s->queries, 0, slots);
    if (s->traced) Vkq().write_timestamp(*cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *s->queries, counters);
    struct ReportSpan { // RT_TRACE_WAITS: per counter type, the reports' record indices
        u32 n = 0, first = 0, last = 0;
    } spans[2];
    double sp_cpu = -1;
    Segments segments(cmd, *s);
    const auto count_samples = [&] {
        SetSampleCounting(cmd, &segments);
        segments.Resume();
    };
    if (q->count_samples) count_samples();
    std::vector<u64> fenced;
    std::vector<std::pair<Queue*, u64>>& waits = waits_io;
    Translator tr{cmd, s->upload, s->keep, &q->render, &s->download, &s->writes, &s->sets, &s->indirect_checks, &s->device};
    s->index.Reset(); // (its writes are empty: cleared at retirement)
    tr.index = &s->index;
    for (const auto& e : q->in_flight) tr.earlier.push_back({&e->writes, false, &e->index}); // stable: retiring needs q->mu
    if (!q->cache_made) q->cache = MakeBufferCache(), q->cache_made = true;
    std::vector<CacheCopy> cache_copies;
    tr.cache = q->cache.get(), tr.cache_copies = &cache_copies;
    tr.submission = s->value, tr.retired = q->retired.load(std::memory_order_acquire);
    s->cache_verify.clear();
    tr.cache_verify = &s->cache_verify;
    s->bytes = {};
    if (s->traced) {
        tr.bytes = &s->bytes;
        tr.time_dispatch = [&, open = false](u64 hash, bool after) mutable {
            const u32 n = static_cast<u32>(s->dispatch_hashes.size());
            if (after) {
                if (open) Vkq().write_timestamp(*cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *s->queries, dispatch_base + 2 * n - 1);
                open = false;
                return;
            }
            open = 2 * n < s->dispatch_slots; // else more vkCmdDispatch than dispatch records: untimed
            if (!open) return;
            Vkq().write_timestamp(*cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *s->queries, dispatch_base + 2 * n);
            s->dispatch_hashes.push_back(hash);
            if (s->heavy.on && !s->heavy.dispatches.empty()) { // the dispatch record being translated (the loop's entry)
                s->heavy.dispatches.back().gpu_slot = static_cast<s32>(n);
                s->heavy.dispatches.back().hash = hash;
            }
        };
    }
    std::map<Op, std::pair<u64, u32>> op_ns; // RT_TRACE_PRESENT, RT_TRACE_WAITS: time and count per record type
    const bool time_ops = TraceSubmits() || s->traced;
    size_t stop = part.size(); // records translated into this part (fewer at a cut)
    for (const Rec& r : part) {
        const u64 t_op = time_ops ? NowNs() : 0;
        struct OpTimer {
            std::map<Op, std::pair<u64, u32>>& m;
            Op op;
            u64 t;
            ~OpTimer() {
                if (!t) return;
                auto& e = m[op];
                e.first += NowNs() - t, ++e.second;
            }
        } op_timer{op_ns, r.op, t_op};
        // RT_PROFILE_HEAVY: the record's kind for the phase timers, the driver time and the dispatch's bytes so far
        const size_t rec_i = static_cast<size_t>(&r - part.data());
        const u8 rec_kind = HeavyKind(r.op);
        u64 driver_ns0 = 0, driver_n0 = 0;
        TraceBytes bytes0{};
        if (s->heavy.on) {
            t_heavy_phases = &s->heavy.phases[rec_kind];
            driver_ns0 = s->heavy.driver.total_ns, driver_n0 = s->heavy.driver.total_n;
            if (rec_kind == 1) {
                bytes0 = s->bytes;
                s->heavy.dispatches.push_back({});
                s->heavy.dispatches.back().record = static_cast<u32>(rec_i);
                s->heavy.dispatches.back().wb_begin = static_cast<u32>(s->writes.size());
            }
        }
        switch (r.op) {
        case Op::kReportCounter:
        case Op::kResetCounter: {
            const auto p = Payload<CounterRecord>(r);
            const bool report = r.op == Op::kReportCounter;
            if (p.counter > 1) Fatal("nvn: %s of counter type %u is not implemented yet (TIMESTAMP and SAMPLES_PASSED only)", OpName(r.op), p.counter);
            if (p.counter == 1 && !q->count_samples) {
                q->count_samples = true;
                count_samples();
            }
            // TIMESTAMP ends the open pass (as before); SAMPLES_PASSED, often between draws, keeps it
            if (p.counter == 0 && report) EndOpenPass(cmd);
            if (q->count_samples) segments.Pause();
            u32 query = 0;
            if (report) {
                for (const CounterEvent& e : s->counters) query += e.report;
                Vkq().write_timestamp(*cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *s->queries, query);
                s->writes.push_back({WriteBack::Kind::kCounter, p.address, 16, nullptr, {}, query});
                if (s->traced) {
                    ReportSpan& sp = spans[p.counter];
                    const u32 at = static_cast<u32>(&r - recs.data());
                    if (!sp.n++) sp.first = at;
                    sp.last = at;
                    if (p.counter == 1) s->last_sample = static_cast<s32>(query), sp_cpu = (NowNs() - t_locked) / 1e6;
                }
            }
            s->counters.push_back({report, p.counter, query, s->segments});
            if (q->count_samples) segments.Resume();
            break;
        }
        case Op::kCopyBufferToBuffer: TranslateCopyBufferToBuffer(tr, Payload<CopyBufferRecord>(r)); break;
        case Op::kBarrier: FullBarrier(cmd); break; // every NVN barrier as a full one, for now
        case Op::kCopyBufferToTexture: {
            Translator& t = tr;
            TranslateCopyBufferToTexture(t, Payload<CopyBufferToTextureRecord>(r));
            break;
        }
        case Op::kClearTexture: {
            Translator& t = tr;
            TranslateClearTexture(t, Payload<ClearTextureRecord>(r));
            break;
        }
        case Op::kSetTexturePool: q->render.texture_pool = Payload<u64>(r); break;
        case Op::kSetSamplerPool: q->render.sampler_pool = Payload<u64>(r); break;
        case Op::kFenceSync: fenced.push_back(Payload<SyncRecord>(r).sync); break;
        case Op::kSignalEvent: {
            const auto e = Payload<EventRecord>(r);
            s->writes.push_back({e.mode == 1 ? WriteBack::Kind::kEventAdd : WriteBack::Kind::kEventWrite, e.address, 4, nullptr, {}, e.value});
            break;
        }
        case Op::kClearBuffer: {
            const auto f = Payload<FillRecord>(r);
            s->writes.push_back({WriteBack::Kind::kFill, f.address, f.size, nullptr, {}, f.value});
            break;
        }
        case Op::kUpdateBuffer: { // inline data written into buffer memory
            u64 address;
            std::memcpy(&address, r.payload.data(), 8);
            std::vector<u8> bytes(r.payload.begin() + 8, r.payload.end());
            const u64 n = bytes.size();
            s->writes.push_back({WriteBack::Kind::kBytes, address, n, nullptr, std::move(bytes), 0});
            break;
        }
        case Op::kCopyTextureToBuffer: TranslateCopyTextureToBuffer(tr, Payload<CopyBufferToTextureRecord>(r)); break;
        case Op::kCopyTextureToTexture: TranslateCopyTextureToTexture(tr, Payload<CopyTextureToTextureRecord>(r)); break;
        case Op::kWaitSync: {
            const u64 sync = Payload<SyncRecord>(r).sync;
            std::lock_guard sl(g_sync_mu);
            const auto* y = static_cast<const Sync*>(FindObject(sync, Type::kSync));
            if (!y) { // finalised (or never initialised): NVN waits on its stale fence, long reached
                WaitOnStaleSync(sync);
                break;
            }
            if (y->host_signaled) break;
            if (!y->queue) Fatal("nvn: a command list waits for sync 0x%llx, which nothing fenced", (unsigned long long)sync);
            if (y->queue != q) waits.push_back({y->queue, y->value}); // in order on its own queue
            break;
        }
        default: {
            Translator& t = tr;
            if (!TranslateRenderRecord(t, r.op, r.payload.data(), r.payload.size()) &&
                !TranslateBindRecord(t, r.op, r.payload.data(), r.payload.size()) &&
                !TranslateDrawRecord(t, r.op, r.payload.data(), r.payload.size())) {
                Fatal("nvn: %s records are not translated yet (NVN layer, RT_GPU=nvn)", OpName(r.op));
            }
        }
        }
        if (s->heavy.on) {
            HeavyProfile& h = s->heavy;
            h.rec_ns[rec_i] = static_cast<u32>(std::min<u64>(NowNs() - t_op, ~0u));
            h.rec_driver_ns[rec_i] = static_cast<u32>(std::min<u64>(h.driver.total_ns - driver_ns0, ~0u));
            h.rec_driver_n[rec_i] = static_cast<u16>(std::min<u64>(h.driver.total_n - driver_n0, 0xffff));
            h.pass_open[rec_i] = PassOpen(cmd);
            if (rec_kind == 1) {
                HeavyDispatch& d = h.dispatches.back();
                const TraceBytes& y = s->bytes;
                d.write_fills = y.write_fills - bytes0.write_fills, d.read_fills = y.read_fills - bytes0.read_fills;
                d.gpu_parts = y.gpu_parts - bytes0.gpu_parts, d.to_device = y.to_device - bytes0.to_device;
                d.from_device = y.from_device - bytes0.from_device, d.writes = y.writes - bytes0.writes;
                d.wb_end = static_cast<u32>(s->writes.size());
            }
            t_heavy_phases = nullptr;
        }
        if (s->traced) {
            const size_t done = static_cast<size_t>(&r - part.data()) + 1;
            if (done % kCheckpointRecords == 0 && done / kCheckpointRecords <= s->checkpoints) {
                Vkq().write_timestamp(*cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *s->queries,
                                      counters + 1 + static_cast<u32>(done / kCheckpointRecords));
                ++s->checkpoints_written;
            }
        }
        // a cut (CutNs): long enough translated, records left, and no render pass open (none is broken)
        if (const u64 cut = CutNs(); cut && &r + 1 != part.data() + part.size() && NowNs() - t_locked >= cut && !PassOpen(cmd)) {
            stop = static_cast<size_t>(&r - part.data()) + 1;
            break;
        }
    }
    // a part cut early: its record lists end there, and the queries it planned for the rest get a value
    if (stop < part.size()) {
        if (s->traced) {
            s->ops.resize(stop);
            u32 reports = 0;
            for (const CounterEvent& e : s->counters) reports += e.report;
            for (u32 i = reports; i < counters; ++i) Vkq().write_timestamp(*cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *s->queries, i);
            for (u32 i = s->checkpoints_written; i < s->checkpoints; ++i) {
                Vkq().write_timestamp(*cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *s->queries, counters + 2 + i);
            }
        }
        if (s->heavy.on) {
            s->heavy.rec_ns.resize(stop), s->heavy.rec_driver_ns.resize(stop), s->heavy.rec_driver_n.resize(stop);
            s->heavy.pass_open.resize(stop);
        }
    }
    EndOpenPass(cmd);
    if (q->count_samples) {
        segments.Pause();
        SetSampleCounting(cmd, nullptr);
    }
    if (s->traced) {
        // dispatch slots left unwritten (skipped or retried dispatches) get one too: every query read has a value
        for (u32 i = 2 * static_cast<u32>(s->dispatch_hashes.size()); i < s->dispatch_slots; ++i) {
            Vkq().write_timestamp(*cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *s->queries, dispatch_base + i);
        }
        Vkq().write_timestamp(*cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *s->queries, counters + 1);
    }
    cmd.End();
    // The buffer cache's page copies this translation asked for, in a command buffer of their own submitted first:
    // after everything earlier on the queue (which may still read the old bytes), before these commands.
    const Vulkan::vk::CommandBuffer copies(s->buffers[1], Vk().device->GetDispatchLoader());
    if (!cache_copies.empty()) {
        copies.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        FullBarrier(copies);
        const VkBuffer arena = CacheBuffer(*q->cache);
        for (const CacheCopy& c : cache_copies) copies.CopyBuffer(c.src, arena, VkBufferCopy{c.src_offset, c.dst_offset, c.size});
        FullBarrier(copies);
        copies.End();
    }
    const std::array<VkCommandBuffer, 2> buffers{s->buffers[1], s->buffers[0]};
    for (size_t i = 0; i < tr.earlier.size(); ++i) {
        if (tr.earlier[i].used) q->in_flight[i]->keep_until = s->value; // its ring is read by this submission
    }
    const u64 t_translated = NowNs();
    s->heavy.translate_ns = t_translated - t_locked;
    std::vector<VkSemaphore> wait_sems;
    std::vector<u64> wait_values;
    std::vector<VkPipelineStageFlags> wait_stages;
    for (auto [other, value] : waits) {
        wait_sems.push_back(*other->timeline);
        wait_values.push_back(value);
        wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    const VkSemaphore signal = *q->timeline;
    const VkTimelineSemaphoreSubmitInfo timeline{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = static_cast<u32>(wait_values.size()),
        .pWaitSemaphoreValues = wait_values.data(),
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &s->value,
    };
    const VkSubmitInfo info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timeline,
        .waitSemaphoreCount = static_cast<u32>(wait_sems.size()),
        .pWaitSemaphores = wait_sems.data(),
        .pWaitDstStageMask = wait_stages.data(),
        .commandBufferCount = cache_copies.empty() ? 1u : 2u,
        .pCommandBuffers = cache_copies.empty() ? cmd.address() : buffers.data(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal,
    };
    std::unique_lock queue_lock(QueueLock());
    if (const VkResult r = Vk().device->GetGraphicsQueue().Submit(info); r != VK_SUCCESS) {
        Fatal("nvn: vkQueueSubmit failed: VkResult %d", r);
    }
    queue_lock.unlock();
    const u64 t_submitted = NowNs();
    s->submitted_ns = TraceSubmits() || s->traced ? t_submitted : 0;
    if (s->traced) {
        const KThread* me = CurrentThread();
        char b[256];
        std::snprintf(b, sizeof b,
                      "submit,%.3f,%.3f,%llu,submit,q%u:%llu,,first=%zu;records=%zu;translate=%.3f;ts=%u@%u-%u;sp=%u@%u-%u;sp_cpu=%.3f",
                      Uptime() * 1e3 - (t_submitted - t0) / 1e6, (t_submitted - t0) / 1e6,
                      (unsigned long long)(me ? me->id : 0), q->trace_id, (unsigned long long)s->value, begin, stop,
                      (t_translated - t_locked) / 1e6, spans[0].n, spans[0].first, spans[0].last, spans[1].n,
                      spans[1].first, spans[1].last, sp_cpu);
        TraceWaitsLine(b);
        TranslateSecond(op_ns, t_translated - t_locked, stop);
    }
    if (TraceSubmits() && t_submitted - t0 > 5'000'000) {
        Log("nvn queue: slow submission (%zu records): %.1f ms for the queue lock, %.1f ms translating, %.1f ms in vkQueueSubmit",
            stop, (t_locked - t0) / 1e6, (t_translated - t_locked) / 1e6, (t_submitted - t_translated) / 1e6);
        std::vector<std::pair<u64, Op>> top;
        for (const auto& [op, e] : op_ns) top.push_back({e.first, op});
        std::sort(top.rbegin(), top.rend());
        for (size_t i = 0; i < top.size() && i < 4; ++i) {
            Log("nvn queue:   %s: %u records, %.1f ms", OpName(top[i].second), op_ns[top[i].second].second, top[i].first / 1e6);
        }
    }
    q->submitted = s->value;
    {
        std::lock_guard sl(g_sync_mu);
        for (u64 sync : fenced) {
            auto* y = static_cast<Sync*>(FindObject(sync, Type::kSync));
            if (!y) continue; // finalised before its fence was submitted: NVN writes stale memory, nobody reads it
            y->queue = q;
            y->value = s->value;
            y->host_signaled = false;
        }
    }
    q->in_flight.push_back(std::move(s));
    q->work.notify_one();
    g_submit_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    g_submits.fetch_add(1, std::memory_order_relaxed);
    return begin + stop;
}

// A submission's counter reports reach guest memory when the Vulkan submission holding them retires, and a
// game may wait for them (Brothership's RenderThread polls its occlusion results each frame, at the start of
// the next frame's work: 6.5 ms a frame, 12-13 in the 60 fps dips, waiting for the whole previous frame).
// So a submission with reports well before its end is cut after the last such one: that part is translated and
// submitted first (the GPU starts on it while the rest is translated) and retires on its own, with the
// reports and the other write-backs before them; its fences signal then (NVN: once the commands before them
// are done). One more render-pass break per such submission. RT_NVN_SPLIT=<records after the last report,
// at least; 256> or 0 (off). Each of the two is cut again wherever its translation runs long (CutNs).
void Submit(GuestContext*, Queue* q, const std::vector<u64>& handles) {
    const u64 t0 = NowNs();
    std::vector<Rec> recs;
    for (u64 h : handles) {
        ForEachRecord(h, [&](Op op, const u8* p, u32 size) { recs.push_back({op, std::vector<u8>(p, p + size)}); });
    }
    size_t split = recs.size();
    if (const size_t min_tail = SplitMinTail()) {
        // the last report with enough records after it (a game may also report at its very end: Brothership
        // ends each frame with a TIMESTAMP, its SAMPLES_PASSED reports sit at ~48 %)
        for (size_t i = recs.size() > min_tail ? recs.size() - min_tail : 0; i-- > 0;) {
            if (recs[i].op != Op::kReportCounter) continue;
            split = i + 1;
            break;
        }
    }
    std::lock_guard lock(q->mu);
    const u64 t_locked = NowNs();
    std::vector<std::pair<Queue*, u64>> waits = std::move(q->waits);
    q->waits.clear();
    // [0, split) and [split, end), each in as many parts as its cuts make (CutNs); an empty submission still
    // goes to the GPU once (it carries the queue's waits)
    if (recs.empty()) {
        SubmitPart(q, recs, 0, 0, waits, t0, t_locked);
        return;
    }
    for (size_t at = 0, part_end = split; at < recs.size(); part_end = recs.size()) {
        bool first = at == 0;
        while (at < part_end) {
            const u64 t1 = first ? t0 : NowNs();
            at = SubmitPart(q, recs, at, part_end, waits, t1, first ? t_locked : t1);
            first = false;
        }
    }
}

void RetireLoop(Queue* q) {
    for (;;) {
        u64 next;
        {
            std::unique_lock lock(q->mu);
            q->work.wait(lock, [&] { return q->stopping || q->submitted > q->retired.load(); });
            if (q->submitted <= q->retired.load()) return; // stopping, nothing left
            next = q->retired.load() + 1;
        }
        q->timeline.Wait(next, 100'000'000);
        std::lock_guard lock(q->mu);
        Retire(q);
    }
}

} // namespace

NVN_PROC(nvnQueueInitialize, bool, u64 queue, u64 builder) {
    const auto b = Load<QueueBuilder>(builder);
    ObjectOf(b.device, Type::kDevice);
    Queue* q = Create<Queue>(queue);
    {
        std::lock_guard ql(g_queues_mu);
        g_queues.insert(q);
    }
    q->device = b.device;
    q->render.device = b.device;
    q->settings = b;
    const auto& logical = Vk().device->GetLogical();
    Vkq();
    const VkSemaphoreTypeCreateInfo type{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    q->timeline = logical.CreateSemaphore({
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &type,
        .flags = 0,
    });
    q->pool = logical.CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = Vk().device->GetGraphicsFamily(),
    });
    q->retirer = std::thread(RetireLoop, q);
    return true;
}
NVN_PROC(nvnQueueFinalize, void, u64 queue) {
    Queue* q = Get<Queue>(queue);
    u64 last;
    {
        std::lock_guard lock(q->mu);
        last = q->submitted;
    }
    WaitRetired(q, last, ~0ULL);
    {
        std::lock_guard lock(q->mu);
        q->stopping = true;
    }
    q->work.notify_one();
    q->retirer.join();
    {
        std::lock_guard ql(g_queues_mu);
        g_queues.erase(q);
    }
    Destroy<Queue>(queue);
}
NVN_PROC(nvnQueueSetDebugLabel, void, u64 queue, u64 label) { Get<Queue>(queue); }
// GetError(queue, info): NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR (0), `info` untouched — the layer's GPU work
// does not fault the way the console's can. Brothership calls it only in its NVN debug callback (BRO 0x31706f8).
NVN_PROC(nvnQueueGetError, u32, u64 queue, u64 info) {
    Get<Queue>(queue);
    return 0;
}

NVN_PROC(nvnQueueSubmitCommands, void, u64 queue, s32 count, u64 handles) {
    Queue* q = Get<Queue>(queue);
    std::vector<u64> h(static_cast<size_t>(std::max(count, 0)));
    if (!h.empty()) g_mem.Read(handles, h.data(), h.size() * sizeof(u64));
    Submit(c, q, h);
}
// The layer submits at once: nothing to flush.
NVN_PROC(nvnQueueFlush, void, u64 queue) {
    Queue* q = Get<Queue>(queue);
    std::lock_guard lock(q->mu);
    Retire(q);
}
NVN_PROC(nvnQueueFinish, void, u64 queue) {
    Queue* q = Get<Queue>(queue);
    u64 last;
    {
        std::lock_guard lock(q->mu);
        last = q->submitted;
    }
    WaitRetired(q, last, ~0ULL);
}
// A sync after everything submitted so far (signaled at once if nothing was).
NVN_PROC(nvnQueueFenceSync, void, u64 queue, u64 sync, u32 condition, u32 flags) {
    Queue* q = Get<Queue>(queue);
    u64 value;
    {
        std::lock_guard lock(q->mu);
        value = q->submitted;
    }
    std::lock_guard sl(g_sync_mu);
    Sync* y = Get<Sync>(sync);
    y->queue = q;
    y->value = value;
    y->host_signaled = false;
}
// The queue's later submissions wait for the sync on the GPU (in order on its own queue).
NVN_PROC(nvnQueueWaitSync, bool, u64 queue, u64 sync) {
    Queue* q = Get<Queue>(queue);
    Queue* other;
    u64 value;
    {
        std::lock_guard sl(g_sync_mu);
        const auto* y = static_cast<const Sync*>(FindObject(sync, Type::kSync));
        if (!y) return WaitOnStaleSync(sync), true;
        if (y->host_signaled) return true;
        if (!y->queue) Fatal("nvn: nvnQueueWaitSync on sync 0x%llx, which nothing fenced", (unsigned long long)sync);
        other = y->queue;
        value = y->value;
    }
    if (other != q) {
        std::lock_guard lock(q->mu);
        q->waits.push_back({other, value});
    }
    return true;
}

// --- For the window (window.cpp) -----------------------------------------------------------------

u64 QueueSubmitted(u64 queue) {
    Queue* q = Get<Queue>(queue);
    std::lock_guard lock(q->mu);
    return q->submitted;
}
void WaitQueue(u64 queue, u64 value) { WaitRetired(Get<Queue>(queue), value, ~0ULL); }
QueueTimes TakeQueueTimes() {
    return {static_cast<double>(g_submit_ns.exchange(0)) / 1e6, static_cast<double>(g_wait_ns.exchange(0)) / 1e6,
            g_submits.exchange(0), static_cast<double>(g_gpu_max_ns.exchange(0)) / 1e6};
}
void SignalSyncOnHost(u64 sync) {
    std::lock_guard sl(g_sync_mu);
    Get<Sync>(sync)->host_signaled = true;
}

// --- Events ---------------------------------------------------------------------------------------
// A word in a memory pool (BRO 0x56cee0...): the builder holds {pool, offset} as NVN's own (16 bytes);
// GetValue reads the word; Signal writes it (mode 0) or adds to it (mode 1) on the CPU; the command
// buffer's SignalEvent does the same when the GPU gets there — here when its submission retires (the
// GPU may write it earlier on the console; never later than its syncs). GetValue retires finished
// submissions first, so a game polling an event sees it change.

namespace {
struct EventBuilder {
    u64 pool;
    s64 offset;
};
struct Event : Object {
    static constexpr Type kType = Type::kEvent;
    u64 pool = 0;
    s64 offset = 0;
    u64 address = 0; // the word
};

void RetireAll() {
    std::lock_guard ql(g_queues_mu);
    for (Queue* q : g_queues) {
        std::lock_guard lock(q->mu);
        Retire(q);
    }
}
} // namespace

NVN_PROC(nvnEventBuilderSetDefaults, void, u64 builder) { Store(builder, EventBuilder{}); }
NVN_PROC(nvnEventBuilderSetStorage, void, u64 builder, u64 pool, s64 offset) { Store(builder, EventBuilder{pool, offset}); }
NVN_PROC(nvnEventBuilderGetStorage, u64, u64 builder, u64 offset_out) {
    const auto b = Load<EventBuilder>(builder);
    hle::WriteGuest<s64>(offset_out, b.offset);
    return b.pool;
}
NVN_PROC(nvnEventBuilderGetMemoryPool, u64, u64 builder) { return Load<EventBuilder>(builder).pool; }
NVN_PROC(nvnEventBuilderGetMemoryOffset, s64, u64 builder) { return Load<EventBuilder>(builder).offset; }

NVN_PROC(nvnEventInitialize, bool, u64 event, u64 builder) {
    const auto b = Load<EventBuilder>(builder);
    const PoolRange p = PoolOf(b.pool);
    if (b.offset < 0 || static_cast<u64>(b.offset) + 4 > p.size) Fatal("nvn: nvnEventInitialize: offset 0x%llx outside its pool", (unsigned long long)b.offset);
    Event* e = Create<Event>(event);
    e->pool = b.pool;
    e->offset = b.offset;
    e->address = p.storage + static_cast<u64>(b.offset);
    return true;
}
NVN_PROC(nvnEventFinalize, void, u64 event) { Destroy<Event>(event); }
NVN_PROC(nvnEventGetMemoryPool, u64, u64 event) { return Get<Event>(event)->pool; }
NVN_PROC(nvnEventGetMemoryOffset, s64, u64 event) { return Get<Event>(event)->offset; }
NVN_PROC(nvnEventGetValue, u32, u64 event) {
    const u64 address = Get<Event>(event)->address;
    RetireAll();
    return hle::ReadGuest<u32>(address);
}
NVN_PROC(nvnEventSignal, void, u64 event, u32 mode, u32 value) {
    ApplyEvent({Get<Event>(event)->address, mode, value});
}
// (cmdbuf, event, mode, location, flags, value): BRO 0x56d090 branches on the 4th argument with values 0
// and 2 (the pipeline location: the GPU may write earlier or later in the pipeline; here after the whole
// submission, which every location allows).
NVN_PROC(nvnCommandBufferSignalEvent, void, u64 cmdbuf, u64 event, u32 mode, u32 location, u32 flags, u32 value) {
    if (mode > 1) Fatal("nvn: nvnCommandBufferSignalEvent mode %u is not implemented yet", mode);
    Record(c, cmdbuf, Op::kSignalEvent, EventRecord{Get<Event>(event)->address, mode, value});
}

// --- Sync ------------------------------------------------------------------------------------------

NVN_PROC(nvnSyncInitialize, bool, u64 sync, u64 device) {
    ObjectOf(device, Type::kDevice);
    std::lock_guard sl(g_sync_mu);
    Create<Sync>(sync);
    return true;
}
NVN_PROC(nvnSyncFinalize, void, u64 sync) {
    std::lock_guard sl(g_sync_mu);
    Destroy<Sync>(sync);
}
NVN_PROC(nvnSyncSetDebugLabel, void, u64 sync, u64 label) { Get<Sync>(sync); }

// NVNsyncWaitResult: 0 already signaled, 1 condition satisfied, 2 timeout
// expired, 3 failed (a sync never fenced, as the SDK answers). Timeout in ns, ~0: forever.
NVN_PROC(nvnSyncWait, u32, u64 sync, u64 timeout_ns) {
    Queue* q;
    u64 value;
    {
        std::lock_guard sl(g_sync_mu);
        const Sync* y = Get<Sync>(sync);
        if (y->host_signaled) return 0;
        q = y->queue;
        value = y->value;
    }
    if (!q) return 3;
    {
        std::lock_guard lock(q->mu);
        Retire(q);
    }
    if (q->retired.load(std::memory_order_acquire) >= value) return 0;
    if (timeout_ns == 0) return 2;
    return WaitRetired(q, value, timeout_ns) ? 1 : 2;
}

// --- Records ---------------------------------------------------------------------------------------

// Control commands in NVN (kinds 5 and 6, 16 bytes).
// ClearBuffer(address, size, value): size 0 records nothing (BRO 0x587ee0). Applied when the submission
// retires; later GPU reads of the range in the same submission see the fill (draws.cpp, ReadAsGpuSees).
NVN_PROC(nvnCommandBufferClearBuffer, void, u64 cmdbuf, u64 address, u64 size, u32 value) {
    if (!size) return;
    if (size % 4 || address % 4) Fatal("nvn: ClearBuffer of 0x%llx bytes at 0x%llx (not whole words)", (unsigned long long)size, (unsigned long long)address);
    const PoolRange p = FindPool(address);
    if (!p.pool || p.offset + size > p.size) Fatal("nvn: ClearBuffer of 0x%llx bytes at 0x%llx: not in one memory pool", (unsigned long long)size, (unsigned long long)address);
    Record(c, cmdbuf, Op::kClearBuffer, FillRecord{address, size, value, 0});
}

// CopyBufferToBuffer(src, dst, size, flags): addresses in memory pools (BRO 0x565fc0: the driver splits
// the copy into 4 MB pieces; the flags are not read there). Translated as a GPU copy whose result reaches
// guest memory at retirement; later GPU reads in the same submission see it (draws.cpp, ReadAsGpuSees).
NVN_PROC(nvnCommandBufferCopyBufferToBuffer, void, u64 cmdbuf, u64 src, u64 dst, u64 size, u32 flags) {
    (void)flags;
    if (!size) return;
    for (const u64 a : {src, dst}) {
        const PoolRange p = FindPool(a);
        if (!p.pool || p.offset + size > p.size) {
            Fatal("nvn: CopyBufferToBuffer of 0x%llx bytes 0x%llx -> 0x%llx: not in one memory pool", (unsigned long long)size,
                  (unsigned long long)src, (unsigned long long)dst);
        }
    }
    Record(c, cmdbuf, Op::kCopyBufferToBuffer, CopyBufferRecord{src, dst, size});
}

// UpdateUniformBuffer(buffer, buffer size, offset, size, data): the GPU writes the data into the buffer in
// order with the other commands; the record keeps a copy (taken when recorded, as NVN copies it into the
// command memory) and the write reaches guest memory at retirement; later GPU reads in the same submission
// see it (draws.cpp, ReadAsGpuSees).
NVN_PROC(nvnCommandBufferUpdateUniformBuffer, void, u64 cmdbuf, u64 buffer, u64 buffer_size, s64 offset, u64 size, u64 data) {
    if (offset < 0 || static_cast<u64>(offset) + size > buffer_size || size > 0xF000) {
        Fatal("nvn: UpdateUniformBuffer of 0x%llx bytes at +0x%llx (buffer 0x%llx bytes)", (unsigned long long)size,
              (unsigned long long)offset, (unsigned long long)buffer_size);
    }
    std::vector<u8> payload(8 + size);
    const u64 address = buffer + static_cast<u64>(offset);
    std::memcpy(payload.data(), &address, 8);
    g_mem.Read(data, payload.data() + 8, size);
    payload.resize((payload.size() + 3) & ~size_t{3});
    Record(c, cmdbuf, Op::kUpdateBuffer, payload.data(), static_cast<u32>(payload.size()));
}

// NVN stores only the sync's address here (BRO 0x55f790 / 0x55f9f0) and reads the sync at submit: the
// game may initialise it after recording, so it is checked then.
NVN_PROC(nvnCommandBufferFenceSync, void, u64 cmdbuf, u64 sync, u32 condition, u32 flags) {
    ControlRecord(c, cmdbuf, Op::kFenceSync, SyncRecord{sync}, 16);
}
NVN_PROC(nvnCommandBufferWaitSync, void, u64 cmdbuf, u64 sync) {
    ControlRecord(c, cmdbuf, Op::kWaitSync, SyncRecord{sync}, 16);
}

} // namespace rt::nvn
