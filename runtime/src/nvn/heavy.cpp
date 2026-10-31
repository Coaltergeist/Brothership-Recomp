// RT_PROFILE_HEAVY (heavy.h): the Vulkan function timers and the three CSV files.
#include "heavy.h"
#include "vk.h"

#include "../threads.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace rt::nvn {

thread_local DriverTimes* t_driver = nullptr;
thread_local HeavyPhases* t_heavy_phases = nullptr;

double HeavyMs() {
    static const double ms = [] {
        const char* e = std::getenv("RT_PROFILE_HEAVY");
        if (!e || !*e || *e == '0') return 0.0;
        const double v = std::atof(e);
        const double at = v == 1 ? 12.0 : v > 0 ? v : 0.0;
        if (at > 0) {
            Log("nvn: heavy submissions (translation or GPU %.1f ms or more) profiled into heavy_*.csv (RT_PROFILE_HEAVY)%s", at,
                TraceWaitsOn() ? "" : "; RT_TRACE_WAITS is off, so nothing is traced: set it too");
        }
        return at;
    }();
    return ms;
}

void HeavyProfile::Reset(bool traced, size_t records) {
    on = traced && HeavyOn();
    translate_ns = t_locked_ns = 0;
    rec_ns.assign(on ? records : 0, 0);
    rec_driver_ns.assign(on ? records : 0, 0);
    rec_driver_n.assign(on ? records : 0, 0);
    pass_open.assign(on ? records : 0, 0);
    phases = {};
    driver = {};
    dispatches.clear();
}

namespace {

u64 Now() {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

const char* g_fn_names[kDriverFns];
u32 g_fn_count = 0;
double g_timer_ns = 0;

/// The wrapper of one table entry: the original function, timed when the calling thread has t_driver.
template <auto Member, typename F>
struct Timed;
template <auto Member, typename R, typename... A>
struct Timed<Member, R(VKAPI_PTR*)(A...)> {
    static inline R(VKAPI_PTR* original)(A...) = nullptr;
    static inline u32 id = 0;
    static R VKAPI_CALL Call(A... a) {
        DriverTimes* d = t_driver;
        if (!d) return original(a...);
        struct Done {
            DriverTimes* d;
            u64 t;
            ~Done() {
                const u64 ns = Now() - t;
                d->ns[id] += ns, ++d->n[id], d->total_ns += ns, ++d->total_n;
            }
        } done{d, Now()};
        return original(a...);
    }
};

template <auto Member, typename F>
void Wrap(F& slot, const char* name) {
    if (!slot || g_fn_count >= kDriverFns) return;
    using T = Timed<Member, F>;
    if (T::original) return; // installed already
    T::original = slot;
    T::id = g_fn_count;
    g_fn_names[g_fn_count++] = name;
    slot = &T::Call;
}

} // namespace

void InstallDriverTimers(Vulkan::vk::DeviceDispatch& d) {
    using D = Vulkan::vk::DeviceDispatch;
#define RT_WRAP(name) Wrap<&D::name>(d.name, #name)
    // every device function the table has (vulkan_wrapper.h's DeviceDispatch), so nothing is left out
    RT_WRAP(vkAcquireNextImageKHR); RT_WRAP(vkAllocateCommandBuffers); RT_WRAP(vkAllocateDescriptorSets);
    RT_WRAP(vkAllocateMemory); RT_WRAP(vkBeginCommandBuffer); RT_WRAP(vkBindBufferMemory); RT_WRAP(vkBindImageMemory);
    RT_WRAP(vkCmdBeginConditionalRenderingEXT); RT_WRAP(vkCmdBeginDebugUtilsLabelEXT); RT_WRAP(vkCmdBeginQuery);
    RT_WRAP(vkCmdBeginRenderPass); RT_WRAP(vkCmdBeginTransformFeedbackEXT); RT_WRAP(vkCmdBindDescriptorSets);
    RT_WRAP(vkCmdBindIndexBuffer); RT_WRAP(vkCmdBindPipeline); RT_WRAP(vkCmdBindTransformFeedbackBuffersEXT);
    RT_WRAP(vkCmdBindVertexBuffers); RT_WRAP(vkCmdBindVertexBuffers2EXT); RT_WRAP(vkCmdBlitImage);
    RT_WRAP(vkCmdClearAttachments); RT_WRAP(vkCmdClearColorImage); RT_WRAP(vkCmdCopyBuffer); RT_WRAP(vkCmdCopyBufferToImage);
    RT_WRAP(vkCmdCopyImage); RT_WRAP(vkCmdCopyImageToBuffer); RT_WRAP(vkCmdCopyQueryPoolResults); RT_WRAP(vkCmdDispatch);
    RT_WRAP(vkCmdDispatchIndirect); RT_WRAP(vkCmdDraw); RT_WRAP(vkCmdDrawIndexed); RT_WRAP(vkCmdDrawIndirect);
    RT_WRAP(vkCmdDrawIndexedIndirect); RT_WRAP(vkCmdDrawIndirectCount); RT_WRAP(vkCmdDrawIndexedIndirectCount);
    RT_WRAP(vkCmdDrawIndirectByteCountEXT); RT_WRAP(vkCmdEndConditionalRenderingEXT); RT_WRAP(vkCmdEndDebugUtilsLabelEXT);
    RT_WRAP(vkCmdEndQuery); RT_WRAP(vkCmdEndRenderPass); RT_WRAP(vkCmdEndTransformFeedbackEXT); RT_WRAP(vkCmdFillBuffer);
    RT_WRAP(vkCmdPipelineBarrier); RT_WRAP(vkCmdPipelineBarrier2); RT_WRAP(vkCmdPushConstants);
    RT_WRAP(vkCmdPushDescriptorSetWithTemplateKHR); RT_WRAP(vkCmdResetQueryPool); RT_WRAP(vkCmdResolveImage);
    RT_WRAP(vkCmdSetBlendConstants); RT_WRAP(vkCmdSetCullModeEXT); RT_WRAP(vkCmdSetDepthBias); RT_WRAP(vkCmdSetDepthBias2EXT);
    RT_WRAP(vkCmdSetDepthBounds); RT_WRAP(vkCmdSetDepthBoundsTestEnableEXT); RT_WRAP(vkCmdSetDepthCompareOpEXT);
    RT_WRAP(vkCmdSetDepthTestEnableEXT); RT_WRAP(vkCmdSetDepthWriteEnableEXT); RT_WRAP(vkCmdSetPrimitiveRestartEnableEXT);
    RT_WRAP(vkCmdSetRasterizerDiscardEnableEXT); RT_WRAP(vkCmdSetAlphaToCoverageEnableEXT); RT_WRAP(vkCmdSetAlphaToOneEnableEXT);
    RT_WRAP(vkCmdSetConservativeRasterizationModeEXT); RT_WRAP(vkCmdSetLineRasterizationModeEXT);
    RT_WRAP(vkCmdSetLineStippleEnableEXT); RT_WRAP(vkCmdSetLineStippleEXT); RT_WRAP(vkCmdSetDepthBiasEnableEXT);
    RT_WRAP(vkCmdSetLogicOpEnableEXT); RT_WRAP(vkCmdSetDepthClampEnableEXT); RT_WRAP(vkCmdSetEvent);
    RT_WRAP(vkCmdSetFrontFaceEXT); RT_WRAP(vkCmdSetPatchControlPointsEXT); RT_WRAP(vkCmdSetLogicOpEXT);
    RT_WRAP(vkCmdSetLineWidth); RT_WRAP(vkCmdSetPrimitiveTopologyEXT); RT_WRAP(vkCmdSetScissor);
    RT_WRAP(vkCmdSetStencilCompareMask); RT_WRAP(vkCmdSetStencilOpEXT); RT_WRAP(vkCmdSetStencilReference);
    RT_WRAP(vkCmdSetStencilTestEnableEXT); RT_WRAP(vkCmdSetStencilWriteMask); RT_WRAP(vkCmdSetVertexInputEXT);
    RT_WRAP(vkCmdSetViewport); RT_WRAP(vkCmdSetColorWriteMaskEXT); RT_WRAP(vkCmdSetColorWriteEnableEXT);
    RT_WRAP(vkCmdSetColorBlendEnableEXT); RT_WRAP(vkCmdSetColorBlendEquationEXT); RT_WRAP(vkCmdWaitEvents);
    RT_WRAP(vkCreateBuffer); RT_WRAP(vkCreateBufferView); RT_WRAP(vkCreateCommandPool); RT_WRAP(vkCreateComputePipelines);
    RT_WRAP(vkCreateDescriptorPool); RT_WRAP(vkCreateDescriptorSetLayout); RT_WRAP(vkCreateDescriptorUpdateTemplate);
    RT_WRAP(vkCreateEvent); RT_WRAP(vkCreateFence); RT_WRAP(vkCreateFramebuffer); RT_WRAP(vkCreateGraphicsPipelines);
    RT_WRAP(vkCreateImage); RT_WRAP(vkCreateImageView); RT_WRAP(vkCreatePipelineCache); RT_WRAP(vkCreatePipelineLayout);
    RT_WRAP(vkCreateQueryPool); RT_WRAP(vkCreateRenderPass); RT_WRAP(vkCreateSampler); RT_WRAP(vkCreateSemaphore);
    RT_WRAP(vkCreateShaderModule); RT_WRAP(vkCreateSwapchainKHR); RT_WRAP(vkDestroyBuffer); RT_WRAP(vkDestroyBufferView);
    RT_WRAP(vkDestroyCommandPool); RT_WRAP(vkDestroyDescriptorPool); RT_WRAP(vkDestroyDescriptorSetLayout);
    RT_WRAP(vkDestroyDescriptorUpdateTemplate); RT_WRAP(vkDestroyEvent); RT_WRAP(vkDestroyFence);
    RT_WRAP(vkDestroyFramebuffer); RT_WRAP(vkDestroyImage); RT_WRAP(vkDestroyImageView); RT_WRAP(vkDestroyPipeline);
    RT_WRAP(vkDestroyPipelineCache); RT_WRAP(vkDestroyPipelineLayout); RT_WRAP(vkDestroyQueryPool);
    RT_WRAP(vkDestroyRenderPass); RT_WRAP(vkDestroySampler); RT_WRAP(vkDestroySemaphore); RT_WRAP(vkDestroyShaderModule);
    RT_WRAP(vkDestroySwapchainKHR); RT_WRAP(vkDeviceWaitIdle); RT_WRAP(vkEndCommandBuffer); RT_WRAP(vkFreeCommandBuffers);
    RT_WRAP(vkFreeDescriptorSets); RT_WRAP(vkFreeMemory); RT_WRAP(vkGetBufferMemoryRequirements2);
    RT_WRAP(vkGetDeviceQueue); RT_WRAP(vkGetEventStatus); RT_WRAP(vkGetFenceStatus); RT_WRAP(vkGetImageMemoryRequirements);
    RT_WRAP(vkGetPipelineCacheData); RT_WRAP(vkGetQueryPoolResults); RT_WRAP(vkGetSemaphoreCounterValue);
    RT_WRAP(vkMapMemory); RT_WRAP(vkQueueSubmit); RT_WRAP(vkQueueSubmit2); RT_WRAP(vkResetFences); RT_WRAP(vkResetQueryPool);
    RT_WRAP(vkUnmapMemory); RT_WRAP(vkUpdateDescriptorSetWithTemplate); RT_WRAP(vkUpdateDescriptorSets);
    RT_WRAP(vkWaitForFences); RT_WRAP(vkWaitSemaphores);
#undef RT_WRAP
    // what one timed call adds by itself (two clock reads), to subtract in the report
    const u64 t0 = Now();
    u64 sink = 0;
    for (int i = 0; i < 100000; ++i) sink += Now();
    g_timer_ns = (Now() - t0) / 100000.0 * 2;
    Log("nvn: %u Vulkan functions timed (RT_PROFILE_HEAVY); a timed call adds ~%.0f ns%s", g_fn_count, g_timer_ns,
        sink ? "" : " ");
}

const char* DriverFnName(u32 id) { return id < g_fn_count ? g_fn_names[id] : "?"; }
u32 DriverFnCount() { return g_fn_count; }
double DriverTimerNs() { return g_timer_ns; }

namespace {

constexpr const char* kKindNames[kHeavyKinds] = {"draw", "dispatch", "other"};

struct Files {
    std::mutex mu;
    FILE* subs = nullptr;
    FILE* stretches = nullptr;
    FILE* dispatches = nullptr;
    // the light submissions of the current second, summed
    u64 second = ~0ull;
    u64 subs_n = 0, records = 0, translate_ns = 0;
    double gpu_ms = 0;
    std::array<u64, kHeavyKinds> kind_ns{}, kind_n{};
    std::array<HeavyPhases, kHeavyKinds> phases{};
    DriverTimes driver;
};

Files& Out() {
    static Files& f = *[] {
        auto* r = new Files; // never freed: written to until the process ends
        r->subs = std::fopen("heavy_subs.csv", "w");
        r->stretches = std::fopen("heavy_stretches.csv", "w");
        r->dispatches = std::fopen("heavy_dispatches.csv", "w");
        if (!r->subs || !r->stretches || !r->dispatches) Fatal("nvn: cannot write heavy_*.csv (RT_PROFILE_HEAVY)");
        std::fputs("sub,t_ms,kind,name,ms,count\n", r->subs);
        std::fputs("sub,stretch,first,last,cpu_end_ms,gpu_ms,pass_open_end,last_safe,draws,dispatches,other\n", r->stretches);
        std::fputs("sub,record,hash,cpu_us,driver_us,driver_calls,gpu_us,writes,write_fills,read_fills,gpu_parts,to_device,"
                   "from_device,wb_bytes,wb_changed,wb_span\n",
                   r->dispatches);
        return r;
    }();
    return f;
}

void Row(FILE* f, const char* sub, double t_ms, const char* kind, const std::string& name, double ms, u64 count) {
    std::fprintf(f, "%s,%.3f,%s,%s,%.4f,%llu\n", sub, t_ms, kind, name.c_str(), ms, (unsigned long long)count);
}

/// The profile's phase, part, kind and driver rows under `sub` (shared by heavy submissions and second sums).
void ProfileRows(FILE* f, const char* sub, double t_ms, const std::array<u64, kHeavyKinds>& kind_ns,
                 const std::array<u64, kHeavyKinds>& kind_n, const std::array<HeavyPhases, kHeavyKinds>& phases,
                 const DriverTimes& driver) {
    for (u32 k = 0; k < kHeavyKinds; ++k) {
        if (kind_n[k]) Row(f, sub, t_ms, "records", kKindNames[k], kind_ns[k] / 1e6, kind_n[k]);
        const std::string phase_kind = std::string("phase.") + kKindNames[k], part_kind = std::string("part.") + kKindNames[k];
        for (u32 p = 0; p < kHeavyPhases; ++p) {
            if (phases[k][p][0]) Row(f, sub, t_ms, phase_kind.c_str(), HeavyPhaseName(p), phases[k][p][0] / 1e6, 0);
            for (u32 q = 0; q < kHeavyParts; ++q) {
                if (!phases[k][p][1 + q]) continue;
                Row(f, sub, t_ms, part_kind.c_str(), std::string(HeavyPhaseName(p)) + "/" + HeavyPartName(q),
                    phases[k][p][1 + q] / 1e6, 0);
            }
        }
    }
    Row(f, sub, t_ms, "sub", "driver_total", driver.total_ns / 1e6, driver.total_n);
    Row(f, sub, t_ms, "sub", "timer_overhead_est", driver.total_n * g_timer_ns / 1e6, driver.total_n);
    for (u32 i = 0; i < g_fn_count; ++i) {
        if (driver.n[i]) Row(f, sub, t_ms, "driver", g_fn_names[i], driver.ns[i] / 1e6, driver.n[i]);
    }
}

void FlushSecond(Files& o) {
    if (!o.subs_n) return;
    char sub[32];
    std::snprintf(sub, sizeof sub, "s%llu", (unsigned long long)o.second);
    const double t_ms = o.second * 1e3;
    Row(o.subs, sub, t_ms, "sub", "submissions", 0, o.subs_n);
    Row(o.subs, sub, t_ms, "sub", "translate", o.translate_ns / 1e6, o.records);
    Row(o.subs, sub, t_ms, "sub", "gpu", o.gpu_ms, 0);
    ProfileRows(o.subs, sub, t_ms, o.kind_ns, o.kind_n, o.phases, o.driver);
    std::fflush(o.subs);
}

} // namespace

void HeavyRetired(u32 queue, u64 value, size_t first_record, double gpu_ms, const HeavyProfile& p,
                  const std::vector<double>& stretch_gpu_ms, u32 stretch_records, const std::vector<double>& dispatch_gpu_ms,
                  const std::vector<u8>& kinds, const std::vector<HeavyWrite>& writes) {
    if (!p.on) return;
    Files& o = Out();
    std::lock_guard lock(o.mu);
    const double translate_ms = p.translate_ns / 1e6;
    const bool heavy = translate_ms >= HeavyMs() || gpu_ms >= HeavyMs();
    std::array<u64, kHeavyKinds> kind_ns{}, kind_n{};
    for (size_t i = 0; i < p.rec_ns.size() && i < kinds.size(); ++i) kind_ns[kinds[i]] += p.rec_ns[i], ++kind_n[kinds[i]];
    const u64 second = static_cast<u64>(Uptime());
    if (second != o.second) {
        FlushSecond(o);
        const u64 keep = second;
        o.subs_n = o.records = o.translate_ns = 0, o.gpu_ms = 0, o.kind_ns = {}, o.kind_n = {}, o.phases = {}, o.driver = {};
        o.second = keep;
    }
    if (!heavy) { // into the second's sums
        ++o.subs_n, o.records += p.rec_ns.size(), o.translate_ns += p.translate_ns, o.gpu_ms += gpu_ms;
        for (u32 k = 0; k < kHeavyKinds; ++k) {
            o.kind_ns[k] += kind_ns[k], o.kind_n[k] += kind_n[k];
            for (u32 a = 0; a < kHeavyPhases; ++a) {
                for (u32 b = 0; b <= kHeavyParts; ++b) o.phases[k][a][b] += p.phases[k][a][b];
            }
        }
        for (u32 i = 0; i < g_fn_count; ++i) o.driver.ns[i] += p.driver.ns[i], o.driver.n[i] += p.driver.n[i];
        o.driver.total_ns += p.driver.total_ns, o.driver.total_n += p.driver.total_n;
        return;
    }
    char sub[48];
    std::snprintf(sub, sizeof sub, "q%u:%llu", queue, (unsigned long long)value);
    const double t_ms = Uptime() * 1e3;
    Row(o.subs, sub, t_ms, "sub", "translate", translate_ms, p.rec_ns.size());
    Row(o.subs, sub, t_ms, "sub", "first_record", 0, first_record);
    Row(o.subs, sub, t_ms, "sub", "gpu", gpu_ms, 0);
    double shaders = 0;
    for (double d : dispatch_gpu_ms) shaders += d;
    Row(o.subs, sub, t_ms, "gpu", "dispatch_shaders", shaders, dispatch_gpu_ms.size());
    Row(o.subs, sub, t_ms, "gpu", "rest", gpu_ms - shaders, 0);
    ProfileRows(o.subs, sub, t_ms, kind_ns, kind_n, p.phases, p.driver);
    // stretches: the CPU time at each one's end (records' own times summed), its GPU time, and where a cut could go
    u64 cpu = 0;
    for (size_t s = 0; s * stretch_records < p.rec_ns.size(); ++s) {
        const size_t a = s * stretch_records, z = std::min(a + stretch_records, p.rec_ns.size()) - 1;
        u32 n[kHeavyKinds]{};
        long long last_safe = -1;
        for (size_t i = a; i <= z; ++i) {
            cpu += p.rec_ns[i];
            if (i < kinds.size()) ++n[kinds[i]];
            if (!p.pass_open[i]) last_safe = static_cast<long long>(first_record + i);
        }
        std::fprintf(o.stretches, "%s,%zu,%zu,%zu,%.4f,%.4f,%u,%lld,%u,%u,%u\n", sub, s, first_record + a, first_record + z, cpu / 1e6,
                     s < stretch_gpu_ms.size() ? stretch_gpu_ms[s] : -1.0, p.pass_open[z], last_safe, n[0], n[1], n[2]);
    }
    for (const HeavyDispatch& d : p.dispatches) {
        u64 wb = 0, changed = 0, span = 0;
        for (u32 i = d.wb_begin; i < d.wb_end && i < writes.size(); ++i) {
            const HeavyWrite& w = writes[i];
            wb += w.size, changed += w.changed;
            if (w.changed) span += w.last - w.first + 1;
        }
        const double gpu_us = d.gpu_slot >= 0 && static_cast<size_t>(d.gpu_slot) < dispatch_gpu_ms.size() ? dispatch_gpu_ms[d.gpu_slot] * 1e3 : -1.0;
        std::fprintf(o.dispatches, "%s,%zu,%016llx,%.2f,%.2f,%u,%.2f,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", sub,
                     first_record + d.record, (unsigned long long)d.hash, p.rec_ns[d.record] / 1e3, p.rec_driver_ns[d.record] / 1e3,
                     p.rec_driver_n[d.record], gpu_us, d.writes, (unsigned long long)d.write_fills, (unsigned long long)d.read_fills,
                     (unsigned long long)d.gpu_parts, (unsigned long long)d.to_device, (unsigned long long)d.from_device,
                     (unsigned long long)wb, (unsigned long long)changed, (unsigned long long)span);
    }
    std::fflush(o.subs), std::fflush(o.stretches), std::fflush(o.dispatches); // the run usually ends by being killed
}

} // namespace rt::nvn
