// Audio: drippu's audio_core (the audio renderer: ADSP, command lists, voices,
// effects) behind HLE audren:u. This file is the host side audio_core is built
// against (runtime/gpu/shim): AudioCore itself (replacing drippu's
// audio_core.cpp, which picks host sinks), a sink, core timing, and the kernel
// shims; then the services (audren:u, hwopus).
//
// Output goes to the host's default device through SDL3 (one SDL stream per
// audout:u session and renderer session, mixed by SDL); the device pulls what
// it plays, which is what paces the renderer (and so the game's audio thread,
// which waits on its event). Without a device (headless runs, RT_AUDIO=0, no
// SDL audio driver) a paced sink drains 240 frames every 5 ms instead.
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <span>
#include <string>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "ipc.h"
#include "kernel.h"
#include "rt.h"
#include "threads.h"

#include "audio_core/audio_core.h"
#include "audio_core/audio_out_manager.h"
#include "audio_core/audio_render_manager.h"
#include "audio_core/common/audio_renderer_parameter.h"
#include "audio_core/out/audio_out.h"
#include "audio_core/out/audio_out_system.h"
#include "audio_core/opus/decoder.h"
#include "audio_core/opus/decoder_manager.h"
#include "audio_core/renderer/audio_device.h"
#include "audio_core/renderer/audio_renderer.h"
#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_stream.h"
#include "common/common_funcs.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_transfer_memory.h"
#include "core/memory.h"
#include "platform/platform.h"

#include <SDL3/SDL.h>

namespace rt::gpu {
Core::System& System(); // gpu.cpp
}

namespace {

using namespace std::chrono_literals;
namespace AS = AudioCore::Sink;

/// A stream with no device behind it: every 5 ms it consumes 240 frames
/// (48 kHz), releasing the renderer's buffers as a real device would.
class PacedStream final : public AS::SinkStream {
public:
    PacedStream(Core::System& system_, AS::StreamType type_) : SinkStream{system_, type_} {
        thread = std::jthread([this](std::stop_token stop) { Run(stop); });
    }
    ~PacedStream() override {
        thread.request_stop();
        if (thread.joinable()) thread.join();
    }

private:
    void Run(std::stop_token stop) {
        constexpr size_t kFrames = 240;
        std::vector<s16> buffer;
        auto next = std::chrono::steady_clock::now();
        while (!stop.stop_requested()) {
            next += 5ms;
            rt::platform::SleepUntil(next);
            buffer.assign(kFrames * GetDeviceChannels(), 0);
            if (type == AS::StreamType::In) {
                ProcessAudioIn(buffer, kFrames);
            } else {
                ProcessAudioOutAndRender(buffer, kFrames);
            }
        }
    }

    std::jthread thread;
};

/// A stream on the host's default playback device (SDL3): SDL's callback pulls
/// what the device needs, so the device's clock paces the game as hardware's
/// does. Several streams (audout:u sessions, renderer sessions) are mixed by SDL.
class SdlStream final : public AS::SinkStream {
public:
    SdlStream(Core::System& system_, AS::StreamType type_, SDL_AudioStream* stream_)
        : SinkStream{system_, type_}, stream{stream_} {
        SDL_SetAudioStreamGetCallback(stream, &SdlStream::Pull, this);
    }
    ~SdlStream() override { SDL_DestroyAudioStream(stream); } // unbinds; no callback after this

    void Start(bool) override {
        if (!paused) return;
        paused = false;
        SDL_ResumeAudioStreamDevice(stream);
    }
    void Stop() override {
        if (paused) return;
        SignalPause();
        SDL_PauseAudioStreamDevice(stream);
    }

private:
    static void Pull(void* self, SDL_AudioStream* s, int additional, int) {
        auto& me = *static_cast<SdlStream*>(self);
        const size_t frames = size_t(additional > 0 ? additional : 0) / sizeof(s16) / me.GetDeviceChannels();
        if (!frames) return;
        me.buffer.assign(frames * me.GetDeviceChannels(), 0);
        me.ProcessAudioOutAndRender(me.buffer, frames);
        SDL_PutAudioStreamData(s, me.buffer.data(), int(me.buffer.size() * sizeof(s16)));
    }

    SDL_AudioStream* stream;
    std::vector<s16> buffer; // only touched on SDL's audio thread
};

/// Host audio: on unless RT_AUDIO=0; headless runs (RT_HEADLESS) are silent unless RT_AUDIO=1.
bool HostAudioWanted() {
    const char* a = std::getenv("RT_AUDIO");
    if (a && *a) return std::strcmp(a, "0") != 0;
    const char* h = std::getenv("RT_HEADLESS");
    return !(h && std::strcmp(h, "0") != 0);
}

/// Output streams go to the host device when one opens; otherwise (or for input)
/// a paced stream, so the game is paced either way.
class HostSink final : public AS::Sink {
public:
    explicit HostSink(bool host_output) : host{host_output && InitHost()} {}

    AS::SinkStream* AcquireSinkStream(Core::System& system, u32, const std::string&,
                                        AS::StreamType type) override {
        std::scoped_lock lock{mutex};
        SDL_AudioStream* s = nullptr;
        if (host && type != AS::StreamType::In) {
            const SDL_AudioSpec spec{SDL_AUDIO_S16, 2, 48000};
            s = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
            if (!s) {
                rt::Log("audio: SDL could not open the playback device (%s); silent from here", SDL_GetError());
                host = false;
            }
        }
        if (s) {
            streams.push_back(std::make_unique<SdlStream>(system, type, s));
        } else {
            streams.push_back(std::make_unique<PacedStream>(system, type));
        }
        return streams.back().get();
    }
    void CloseStream(AS::SinkStream* stream) override {
        std::scoped_lock lock{mutex};
        std::erase_if(streams, [&](const auto& s) { return s.get() == stream; });
    }
    void CloseStreams() override {
        std::scoped_lock lock{mutex};
        streams.clear();
    }
    f32 GetDeviceVolume() const override { return 1.0f; }
    void SetDeviceVolume(f32) override {}
    void SetSystemVolume(f32) override {}

private:
    static bool InitHost() {
        // The hardware's period (240 frames, 5 ms); the driver may round it up.
        SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, "240");
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            rt::Log("audio: no host audio (SDL: %s); silent", SDL_GetError());
            return false;
        }
        rt::Log("audio: host output through SDL, driver %s", SDL_GetCurrentAudioDriver());
        return true;
    }

    bool host;
    std::mutex mutex;
    std::vector<std::unique_ptr<AS::SinkStream>> streams;
};

/// Looping core-timing events (audio in/out sessions), each on its own host thread.
struct LoopingEvents {
    std::mutex mutex;
    std::map<Core::Timing::EventType*, std::jthread> threads;
};
LoopingEvents& Loops() {
    static LoopingEvents l;
    return l;
}

const auto g_start = std::chrono::steady_clock::now();
Kernel::KernelCore g_kernel;
Kernel::KProcess g_process;

} // namespace

// --- Shim implementations ------------------------------------------------------

namespace AudioCore {

AudioCore::AudioCore(Core::System& system) {
    audio_manager.emplace();
    output_sink = std::make_unique<HostSink>(HostAudioWanted());
    input_sink = std::make_unique<HostSink>(false);
    adsp.emplace(system, *output_sink); // after the sinks
}
AudioCore::~AudioCore() {
    Shutdown();
}
void AudioCore::CreateSinks() {}
void AudioCore::Shutdown() {
    audio_manager->Shutdown();
}
AudioManager& AudioCore::GetAudioManager() {
    return *audio_manager;
}
AS::Sink& AudioCore::GetOutputSink() {
    return *output_sink;
}
AS::Sink& AudioCore::GetInputSink() {
    return *input_sink;
}
ADSP::ADSP& AudioCore::ADSP() {
    return *adsp;
}

} // namespace AudioCore

namespace Core {

AudioCore::AudioCore& System::AudioCore() {
    static ::AudioCore::AudioCore audio{*this};
    return audio;
}
Kernel::KernelCore& System::Kernel() {
    return g_kernel;
}

namespace Timing {

std::chrono::nanoseconds CoreTiming::GetGlobalTimeNs() const {
    return std::chrono::steady_clock::now() - g_start;
}
std::chrono::microseconds CoreTiming::GetGlobalTimeUs() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(GetGlobalTimeNs());
}

std::shared_ptr<EventType> CreateEvent(std::string name, TimedCallback&& callback) {
    return std::make_shared<EventType>(std::move(callback), std::move(name));
}

void CoreTiming::ScheduleLoopingEvent(std::chrono::nanoseconds start_time, std::chrono::nanoseconds resched_time,
                                      const std::shared_ptr<EventType>& event_type, bool) {
    auto& loops = Loops();
    std::scoped_lock lock{loops.mutex};
    loops.threads[event_type.get()] = std::jthread([this, start_time, resched_time, event = event_type](
                                                       std::stop_token stop) {
        auto next = std::chrono::steady_clock::now() + start_time;
        while (!stop.stop_requested()) {
            rt::platform::SleepUntil(next);
            if (stop.stop_requested()) break;
            const auto late = std::chrono::steady_clock::now() - next;
            event->callback(GetGlobalTimeNs().count(), late);
            next += resched_time;
        }
    });
}

void CoreTiming::UnscheduleEvent(const std::shared_ptr<EventType>& event_type, UnscheduleEventType) {
    std::jthread thread;
    {
        auto& loops = Loops();
        std::scoped_lock lock{loops.mutex};
        auto it = loops.threads.find(event_type.get());
        if (it == loops.threads.end()) return;
        thread = std::move(it->second);
        loops.threads.erase(it);
    }
    // jthread's destructor stops and joins (outside the map lock: the callback may schedule).
}

} // namespace Timing

namespace Memory {

bool Memory::ZeroBlock(Common::ProcessAddress dest, std::size_t size) {
    std::memset(rt::g_mem.Raw(GetInteger(dest)), 0, size);
    return true;
}

} // namespace Memory
} // namespace Core

Core::Memory::Memory& Kernel::KProcess::GetMemory() {
    return Core::System::GetInstance().ApplicationMemory();
}

// --- Services --------------------------------------------------------------------

namespace rt {
namespace {

constexpr u32 kResultAudioOutOfSessions = 0x0A99; // audio module 153, description 5

/// A Kernel::KEvent (as audio_core signals it) backed by a guest-visible event.
struct BridgedEvent {
    explicit BridgedEvent(const char* name) {
        guest->name = name;
        core.set = [e = guest](bool signaled) {
            // May run on audio_core's threads, inside its own locks, while a guest
            // thread holding the global lock waits for those: never take that lock here.
            e->signaled = signaled;
            if (signaled) WakeWaiters();
        };
    }
    std::shared_ptr<KEvent> guest = std::make_shared<KEvent>();
    Kernel::KEvent core;
};

AudioCore::Renderer::Manager& RenderManager() {
    static AudioCore::Renderer::Manager manager{gpu::System()};
    return manager;
}

class AudioRenderer : public Service {
    s32 session_id_;
    std::string last_voices_; // RT_AUDIO_VOICES

public:
    AudioRenderer(const AudioCore::AudioRendererParameterInternal& params, std::shared_ptr<KTransferMemory> tmem,
                  u64 aruid, s32 session_id)
        : Service("audren:IAudioRenderer"), session_id_{session_id}, tmem_{std::move(tmem)},
          core_tmem_{tmem_->address, tmem_->size},
          impl_{gpu::System(), RenderManager(), &event_.core} {
        const auto result = impl_.Initialize(params, &core_tmem_, tmem_->size, &g_process, aruid, session_id);
        if (result.IsError()) {
            Fatal("audren: renderer initialisation failed (0x%x)", result.raw);
        }
        Log("audren renderer %d: %u Hz, %u samples, %u voices, %u mixes, %u sinks, %u effects, revision %c%c%c%c",
            session_id, params.sample_rate, params.sample_count, params.voices, params.mixes, params.sinks,
            params.effects, params.revision & 0xFF, (params.revision >> 8) & 0xFF, (params.revision >> 16) & 0xFF,
            params.revision >> 24);
        auto& sys = impl_.GetSystem();
        Register(0, "GetSampleRate", [&sys](IpcContext& ctx) { ctx.Push<u32>(sys.GetSampleRate()); });
        Register(1, "GetSampleCount", [&sys](IpcContext& ctx) { ctx.Push<u32>(sys.GetSampleCount()); });
        Register(2, "GetMixBufferCount", [&sys](IpcContext& ctx) { ctx.Push<u32>(sys.GetMixBufferCount()); });
        Register(3, "GetState", [&sys](IpcContext& ctx) { ctx.Push<u32>(sys.IsActive() ? 0 : 1); });
        Register(4, "RequestUpdate", [this](IpcContext& ctx) { RequestUpdate(ctx); });
        Register(5, "Start", [this](IpcContext&) { impl_.Start(); });
        Register(6, "Stop", [this](IpcContext&) { impl_.Stop(); });
        Register(7, "QuerySystemEvent", [this](IpcContext& ctx) {
            ctx.out_copy.push_back(g_handles.Add(event_.guest));
        });
        Register(8, "SetRenderingTimeLimit", [&sys](IpcContext& ctx) { sys.SetRenderingTimeLimit(ctx.Pop<u32>()); });
        Register(9, "GetRenderingTimeLimit", [&sys](IpcContext& ctx) { ctx.Push<u32>(sys.GetRenderingTimeLimit()); });
        Register(10, "RequestUpdateAuto", [this](IpcContext& ctx) { RequestUpdate(ctx); });
        Register(12, "SetVoiceDropParameter", [&sys](IpcContext& ctx) { sys.SetVoiceDropParameter(ctx.Pop<f32>()); });
        Register(13, "GetVoiceDropParameter", [&sys](IpcContext& ctx) { ctx.Push<f32>(sys.GetVoiceDropParameter()); });
    }
    ~AudioRenderer() override { impl_.Finalize(); }

private:
    void RequestUpdate(IpcContext& ctx) {
        const IpcBuffer in = ctx.In(0), out = ctx.Out(0), perf = ctx.Out(1);
        const std::span<const u8> input{g_mem.Raw(in.address), in.size};
        const std::span<u8> output{g_mem.Raw(out.address), out.size};
        const std::span<u8> performance{perf.size ? g_mem.Raw(perf.address) : nullptr, perf.size};
        LogVoices(input);
        const auto result = impl_.RequestUpdate(input, performance, output);
        if (result.IsError()) {
            Log("audren RequestUpdate failed: 0x%x", result.raw);
        }
        ctx.result = result.raw;
    }

    // RT_AUDIO_VOICES=1: the voices the game has in use on this renderer (from each update's voice section:
    // header 0x40 bytes, then behaviour, memory pools, voice channel resources, voices of 0x170 bytes each,
    // as drippu's System::Update reads them), logged whenever that set changes: count, and per voice its
    // sample format (nn::audio::SampleFormat), rate and channel count.
    void LogVoices(std::span<const u8> in) {
        static const bool on = [] {
            const char* e = std::getenv("RT_AUDIO_VOICES");
            return e && *e && *e != '0';
        }();
        if (!on || in.size() < 0x40) return;
        u32 h[11];
        std::memcpy(h, in.data(), sizeof h); // revision, behaviour, memory pools, voices, voice resources ...
        const size_t at = 0x40 + size_t{h[1]} + h[2] + h[4], n = h[3] / 0x170;
        if (at + size_t{h[3]} > in.size()) return;
        std::string desc;
        u32 used = 0;
        for (size_t i = 0; i < n; ++i) {
            const u8* v = in.data() + at + i * 0x170;
            if (!v[9]) continue; // in_use
            u32 rate, channels;
            std::memcpy(&rate, v + 0xc, 4);
            std::memcpy(&channels, v + 0x18, 4);
            char b[48];
            std::snprintf(b, sizeof b, "%s[format %u, %u Hz, %u ch]", used ? " " : ": ", v[0xb], rate, channels);
            desc += b;
            ++used;
        }
        if (desc == last_voices_) return;
        last_voices_ = desc;
        Log("audren renderer %d (t=%.1f s): %u of %zu voices in use%s", session_id_, Uptime(), used, n,
            desc.size() > 600 ? (desc.substr(0, 600) + " ...").c_str() : desc.c_str());
    }

    std::shared_ptr<KTransferMemory> tmem_;
    Kernel::KTransferMemory core_tmem_;
    BridgedEvent event_{"audio renderer"};
    AudioCore::Renderer::Renderer impl_;
};

class AudioDevice : public Service {
public:
    AudioDevice(u64 aruid, u32 revision)
        : Service("audren:IAudioDevice"), impl_{gpu::System(), aruid, revision} {
        using Name = AudioCore::Renderer::AudioDevice::AudioDeviceName;
        auto list = [this](IpcContext& ctx, bool output_only) {
            const IpcBuffer out = ctx.Out(0);
            std::vector<Name> names(out.size / sizeof(Name), Name{""});
            const u32 n = output_only ? impl_.ListAudioOutputDeviceName(names) : impl_.ListAudioDeviceName(names);
            ctx.WriteOut(0, names.data(), n * sizeof(Name));
            ctx.Push<s32>(static_cast<s32>(n));
        };
        auto active = [](IpcContext& ctx) {
            const Name name{"AudioTvOutput"};
            ctx.WriteOut(0, &name, sizeof name);
        };
        auto set_volume = [this](IpcContext& ctx) {
            const f32 volume = ctx.Pop<f32>();
            if (ctx.ReadInString(0) == "AudioTvOutput") impl_.SetDeviceVolumes(volume);
        };
        auto get_volume = [this](IpcContext& ctx) {
            const std::string name = ctx.ReadInString(0);
            ctx.Push<f32>(name == "AudioTvOutput" ? impl_.GetDeviceVolume(name) : 1.0f);
        };
        auto event = [this](IpcContext& ctx) { ctx.out_copy.push_back(g_handles.Add(event_)); };
        Register(0, "ListAudioDeviceName", [list](IpcContext& ctx) { list(ctx, false); });
        Register(1, "SetAudioDeviceOutputVolume", set_volume);
        Register(2, "GetAudioDeviceOutputVolume", get_volume);
        Register(3, "GetActiveAudioDeviceName", active);
        Register(4, "QueryAudioDeviceSystemEvent", [this](IpcContext& ctx) {
            event_->signaled = true; // as drippu does: the device list is "new"
            ctx.out_copy.push_back(g_handles.Add(event_));
        });
        Register(5, "GetActiveChannelCount", [](IpcContext& ctx) { ctx.Push<u32>(2); });
        Register(6, "ListAudioDeviceNameAuto", [list](IpcContext& ctx) { list(ctx, false); });
        Register(7, "SetAudioDeviceOutputVolumeAuto", set_volume);
        Register(8, "GetAudioDeviceOutputVolumeAuto", get_volume);
        Register(10, "GetActiveAudioDeviceNameAuto", active);
        Register(11, "QueryAudioDeviceInputEvent", event);
        Register(12, "QueryAudioDeviceOutputEvent", event);
        Register(13, "GetActiveAudioOutputDeviceName", active);
        Register(14, "ListAudioOutputDeviceName", [list](IpcContext& ctx) { list(ctx, true); });
        Register(19, "SetAudioDeviceOutputVolumeAutoTuneEnabled", [](IpcContext&) {});
        Register(20, "IsAudioDeviceOutputVolumeAutoTuneEnabled", [](IpcContext& ctx) { ctx.Push<u8>(0); });
        event_->name = "audio device";
    }

private:
    AudioCore::Renderer::AudioDevice impl_;
    std::shared_ptr<KEvent> event_ = std::make_shared<KEvent>();
};

class AudioRendererManager : public Service {
public:
    AudioRendererManager() : Service("audren:u") {
        Register(0, "OpenAudioRenderer", [](IpcContext& ctx) {
            const auto params = ctx.Pop<AudioCore::AudioRendererParameterInternal>();
            const u64 tmem_size = ctx.Pop<u64>();
            const u64 aruid = ctx.Pop<u64>();
            auto tmem = ctx.copy_handles.empty() ? nullptr : g_handles.GetAs<KTransferMemory>(ctx.copy_handles[0]);
            if (!tmem) Fatal("audren:u OpenAudioRenderer: no transfer memory handle");
            if (tmem->size < tmem_size) Fatal("audren:u OpenAudioRenderer: transfer memory smaller than its size argument");
            auto& manager = RenderManager();
            const s32 session = manager.GetSessionCount() + 1 > AudioCore::MaxRendererSessions ? -1 : manager.GetSessionId();
            if (session < 0) {
                ctx.result = kResultAudioOutOfSessions;
                return;
            }
            ctx.PushObject(std::make_shared<AudioRenderer>(params, tmem, aruid, session));
        });
        Register(1, "GetWorkBufferSize", [](IpcContext& ctx) {
            const auto params = ctx.Pop<AudioCore::AudioRendererParameterInternal>();
            u64 size = 0;
            const auto result = RenderManager().GetWorkBufferSize(params, size);
            ctx.result = result.raw;
            ctx.Push<u64>(size);
        });
        Register(2, "GetAudioDeviceService", [](IpcContext& ctx) {
            const u64 aruid = ctx.Pop<u64>();
            ctx.PushObject(std::make_shared<AudioDevice>(aruid, Common::MakeMagic('R', 'E', 'V', '1')));
        });
        Register(4, "GetAudioDeviceServiceWithRevisionInfo", [](IpcContext& ctx) {
            const u32 revision = ctx.Pop<u32>();
            const u64 aruid = ctx.Pop<u64>();
            ctx.PushObject(std::make_shared<AudioDevice>(aruid, revision));
        });
    }
};

// --- audout:u: audio_core's AudioOut sessions (PCM buffers straight to the sink) ---

namespace AO = AudioCore::AudioOut;

AO::Manager& OutManager() {
    static AO::Manager manager{gpu::System()};
    return manager;
}

class AudioOut : public Service {
public:
    explicit AudioOut(size_t session_id)
        : Service("audout:IAudioOut"), impl_{std::make_shared<AO::Out>(gpu::System(), OutManager(), &event_.core, session_id)} {
        auto append = [this](IpcContext& ctx) {
            const u64 tag = ctx.Pop<u64>();
            const IpcBuffer in = ctx.In(0);
            if (in.size < sizeof(AO::AudioOutBuffer)) {
                ctx.result = 0x2A99; // audio: InsufficientBuffer (as drippu)
                return;
            }
            AO::AudioOutBuffer buffer;
            g_mem.Read(in.address, &buffer, sizeof buffer);
            ctx.result = impl_->AppendBuffer(buffer, tag).raw;
        };
        auto released = [this](IpcContext& ctx) {
            const IpcBuffer out = ctx.Out(0);
            std::vector<u64> tags(out.size / sizeof(u64));
            if (!tags.empty()) tags[0] = 0;
            const u32 count = impl_->GetReleasedBuffers(tags);
            ctx.WriteOut(0, tags.data(), tags.size() * sizeof(u64));
            ctx.Push<u32>(count);
        };
        Register(0, "GetAudioOutState", [this](IpcContext& ctx) { ctx.Push<u32>(static_cast<u32>(impl_->GetState())); });
        Register(1, "Start", [this](IpcContext& ctx) { ctx.result = impl_->StartSystem().raw; });
        Register(2, "Stop", [this](IpcContext& ctx) { ctx.result = impl_->StopSystem().raw; });
        Register(3, "AppendAudioOutBuffer", append);
        Register(4, "RegisterBufferEvent", [this](IpcContext& ctx) { ctx.out_copy.push_back(g_handles.Add(event_.guest)); });
        Register(5, "GetReleasedAudioOutBuffers", released);
        Register(6, "ContainsAudioOutBuffer", [this](IpcContext& ctx) { ctx.Push<u8>(impl_->ContainsAudioBuffer(ctx.Pop<u64>())); });
        Register(7, "AppendAudioOutBufferAuto", append);
        Register(8, "GetReleasedAudioOutBuffersAuto", released);
        Register(9, "GetAudioOutBufferCount", [this](IpcContext& ctx) { ctx.Push<u32>(impl_->GetBufferCount()); });
        Register(10, "GetAudioOutPlayedSampleCount", [this](IpcContext& ctx) { ctx.Push<u64>(impl_->GetPlayedSampleCount()); });
        Register(11, "FlushAudioOutBuffers", [this](IpcContext& ctx) { ctx.Push<u8>(impl_->FlushAudioOutBuffers()); });
        Register(12, "SetAudioOutVolume", [this](IpcContext& ctx) { impl_->SetVolume(ctx.Pop<f32>()); });
        Register(13, "GetAudioOutVolume", [this](IpcContext& ctx) { ctx.Push<f32>(impl_->GetVolume()); });
    }
    ~AudioOut() override { impl_->Free(); }

    const std::shared_ptr<AO::Out>& Impl() const { return impl_; }

private:
    BridgedEvent event_{"audio out buffer"};
    std::shared_ptr<AO::Out> impl_;
};

class AudioOutManager : public Service {
public:
    AudioOutManager() : Service("audout:u") {
        auto list = [](IpcContext& ctx) {
            const IpcBuffer out = ctx.Out(0);
            const AudioCore::Renderer::AudioDevice::AudioDeviceName name{"DeviceOut"};
            const bool room = out.size >= sizeof name;
            if (room) ctx.WriteOut(0, &name, sizeof name);
            ctx.Push<u32>(room ? 1 : 0);
        };
        auto open = [](IpcContext& ctx) {
            const auto params = ctx.Pop<AO::AudioOutParameter>();
            const u64 aruid = ctx.Pop<u64>();
            auto& manager = OutManager();
            size_t session_id = 0;
            if (auto r = manager.LinkToManager(); r.IsError()) {
                ctx.result = r.raw;
                return;
            }
            if (auto r = manager.AcquireSessionId(session_id); r.IsError()) {
                ctx.result = r.raw;
                return;
            }
            auto out = std::make_shared<AudioOut>(session_id);
            auto& system = out->Impl()->GetSystem();
            // The process handle is the client itself; audio_core reads its buffers through g_process.
            if (auto r = system.Initialize(ctx.ReadInString(0), params, &g_process, aruid); r.IsError()) {
                ctx.result = r.raw;
                return;
            }
            manager.sessions[session_id] = out->Impl();
            manager.applet_resource_user_ids[session_id] = aruid;
            const AudioCore::Renderer::AudioDevice::AudioDeviceName name{system.GetName().c_str()};
            ctx.WriteOut(0, &name, std::min<size_t>(sizeof name, ctx.Out(0).size));
            ctx.Push(AO::AudioOutParameterInternal{.sample_rate = system.GetSampleRate(),
                                                   .channel_count = system.GetChannelCount(),
                                                   .sample_format = static_cast<u32>(system.GetSampleFormat()),
                                                   .state = static_cast<u32>(system.GetState())});
            Log("audout session %zu: %s, %u Hz, %u channels", session_id, system.GetName().c_str(),
                system.GetSampleRate(), system.GetChannelCount());
            ctx.PushObject(out);
        };
        Register(0, "ListAudioOuts", list);
        Register(1, "OpenAudioOut", open);
        Register(2, "ListAudioOutsAuto", list);
        Register(3, "OpenAudioOutAuto", open);
    }
};

// --- hwopus: audio_core's Opus decoder (libopus) ---

namespace Opus = AudioCore::OpusDecoder;

Opus::OpusDecoderManager& OpusManager() {
    static Opus::OpusDecoderManager manager{gpu::System()};
    return manager;
}

class HardwareOpusDecoder : public Service {
public:
    HardwareOpusDecoder(std::shared_ptr<KTransferMemory> tmem)
        : Service("hwopus:IHardwareOpusDecoder"), tmem_{std::move(tmem)}, core_tmem_{tmem_->address, tmem_->size},
          impl_{gpu::System(), OpusManager().GetHardwareOpus()} {
        // Old forms: no reset flag; "WithPerf" forms also return the time taken.
        auto decode = [this](bool multi, bool perf, bool has_reset) {
            return [this, multi, perf, has_reset](IpcContext& ctx) {
                const bool reset = has_reset && ctx.Pop<u8>() != 0;
                const IpcBuffer in = ctx.In(0), out = ctx.Out(0);
                u32 data_size = 0, samples = 0;
                u64 time = 0;
                const std::span<const u8> input{g_mem.Raw(in.address), in.size};
                const std::span<u8> output{g_mem.Raw(out.address), out.size};
                const auto r = multi ? impl_.DecodeInterleavedForMultiStream(&data_size, &time, &samples, input, output, reset)
                                     : impl_.DecodeInterleaved(&data_size, &time, &samples, input, output, reset);
                ctx.result = r.raw;
                ctx.Push<u32>(data_size);
                ctx.Push<u32>(samples);
                if (perf) ctx.Push<u64>(time);
            };
        };
        auto set_context = [this](IpcContext& ctx) {
            const IpcBuffer in = ctx.In(0);
            ctx.result = impl_.SetContext({g_mem.Raw(in.address), in.size}).raw;
        };
        Register(0, "DecodeInterleavedOld", decode(false, false, false));
        Register(1, "SetContext", set_context);
        Register(2, "DecodeInterleavedForMultiStreamOld", decode(true, false, false));
        Register(3, "SetContextForMultiStream", set_context);
        Register(4, "DecodeInterleavedWithPerfOld", decode(false, true, false));
        Register(5, "DecodeInterleavedForMultiStreamWithPerfOld", decode(true, true, false));
        Register(6, "DecodeInterleavedWithPerfAndResetOld", decode(false, true, true));
        Register(7, "DecodeInterleavedForMultiStreamWithPerfAndResetOld", decode(true, true, true));
        Register(8, "DecodeInterleaved", decode(false, true, true));
        Register(9, "DecodeInterleavedForMultiStream", decode(true, true, true));
    }

    template <typename Params>
    u32 Initialize(const Params& params) {
        return impl_.Initialize(params, &core_tmem_, tmem_->size).raw;
    }

private:
    std::shared_ptr<KTransferMemory> tmem_;
    Kernel::KTransferMemory core_tmem_;
    Opus::OpusDecoder impl_;
};

class HardwareOpusDecoderManager : public Service {
public:
    HardwareOpusDecoderManager() : Service("hwopus") {
        // Single-stream parameters arrive in the raw data; multi-stream ones in a pointer (X) buffer.
        auto open = [](auto params, IpcContext& ctx) {
            const u32 tmem_size = ctx.Pop<u32>();
            auto tmem = ctx.copy_handles.empty() ? nullptr : g_handles.GetAs<KTransferMemory>(ctx.copy_handles[0]);
            if (!tmem || tmem->size < tmem_size) Fatal("hwopus: bad transfer memory");
            auto decoder = std::make_shared<HardwareOpusDecoder>(tmem);
            ctx.result = decoder->Initialize(params);
            if (ctx.result == kResultSuccess) ctx.PushObject(decoder);
        };
        auto from_pointer = []<typename T>(IpcContext& ctx, T& out) {
            const IpcBuffer in = ctx.In(0);
            if (in.size < sizeof(T)) Fatal("hwopus: parameter buffer too small (0x%llx)", (unsigned long long)in.size);
            std::memcpy(&out, g_mem.Raw(in.address), sizeof(T));
        };
        Register(0, "OpenHardwareOpusDecoder", [open](IpcContext& ctx) {
            const auto p = ctx.Pop<Opus::OpusParameters>();
            Opus::OpusParametersEx ex{};
            ex.sample_rate = p.sample_rate;
            ex.channel_count = p.channel_count;
            open(ex, ctx);
        });
        Register(1, "GetWorkBufferSize", [](IpcContext& ctx) {
            u32 size = 0;
            ctx.result = OpusManager().GetWorkBufferSize(ctx.Pop<Opus::OpusParameters>(), size).raw;
            ctx.Push<u32>(size);
        });
        Register(2, "OpenOpusDecoderForMultiStream", [open, from_pointer](IpcContext& ctx) {
            Opus::OpusMultiStreamParameters p;
            from_pointer(ctx, p);
            Opus::OpusMultiStreamParametersEx ex{};
            ex.sample_rate = p.sample_rate;
            ex.channel_count = p.channel_count;
            ex.total_stream_count = p.total_stream_count;
            ex.stereo_stream_count = p.stereo_stream_count;
            std::memcpy(ex.mappings.data(), p.mappings.data(), sizeof p.mappings);
            open(ex, ctx);
        });
        Register(3, "GetWorkBufferSizeForMultiStream", [from_pointer](IpcContext& ctx) {
            Opus::OpusMultiStreamParameters p;
            from_pointer(ctx, p);
            u32 size = 0;
            ctx.result = OpusManager().GetWorkBufferSizeForMultiStream(p, size).raw;
            ctx.Push<u32>(size);
        });
        Register(4, "OpenHardwareOpusDecoderEx", [open](IpcContext& ctx) {
            open(ctx.Pop<Opus::OpusParametersEx>(), ctx);
        });
        Register(5, "GetWorkBufferSizeEx", [](IpcContext& ctx) {
            u32 size = 0;
            ctx.result = OpusManager().GetWorkBufferSizeEx(ctx.Pop<Opus::OpusParametersEx>(), size).raw;
            ctx.Push<u32>(size);
        });
        Register(6, "OpenHardwareOpusDecoderForMultiStreamEx", [open, from_pointer](IpcContext& ctx) {
            Opus::OpusMultiStreamParametersEx p;
            from_pointer(ctx, p);
            open(p, ctx);
        });
        Register(7, "GetWorkBufferSizeForMultiStreamEx", [from_pointer](IpcContext& ctx) {
            Opus::OpusMultiStreamParametersEx p;
            from_pointer(ctx, p);
            u32 size = 0;
            ctx.result = OpusManager().GetWorkBufferSizeForMultiStreamEx(p, size).raw;
            ctx.Push<u32>(size);
        });
        Register(8, "GetWorkBufferSizeExEx", [](IpcContext& ctx) {
            u32 size = 0;
            ctx.result = OpusManager().GetWorkBufferSizeExEx(ctx.Pop<Opus::OpusParametersEx>(), size).raw;
            ctx.Push<u32>(size);
        });
        Register(9, "GetWorkBufferSizeForMultiStreamExEx", [from_pointer](IpcContext& ctx) {
            Opus::OpusMultiStreamParametersEx p;
            from_pointer(ctx, p);
            u32 size = 0;
            ctx.result = OpusManager().GetWorkBufferSizeForMultiStreamExEx(p, size).raw;
            ctx.Push<u32>(size);
        });
    }
};

const bool registered = [] {
    RegisterServiceFactory("audren:u", [] { return std::make_shared<AudioRendererManager>(); });
    RegisterServiceFactory("audout:u", [] { return std::make_shared<AudioOutManager>(); });
    RegisterServiceFactory("hwopus", [] { return std::make_shared<HardwareOpusDecoderManager>(); });
    return true;
}();

} // namespace
} // namespace rt
