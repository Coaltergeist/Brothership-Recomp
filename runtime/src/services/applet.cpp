// am (appletOE) for an application, and apm (performance management).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>

#include "../ipc.h"
#include "../options.h"
#include "../services.h"

namespace rt {
namespace {

constexpr u32 kResultNoMessage = 0x680; // am module 128, description 3

/// Applet state shared by the proxy's sub-objects.
constexpr u32 kMessageFocusStateChanged = 15;

struct AppletState {
    std::deque<u32> messages;
    std::shared_ptr<KEvent> message_event = std::make_shared<KEvent>();
    std::shared_ptr<KEvent> suspended_tick_event = std::make_shared<KEvent>();
    std::shared_ptr<KEvent> gpu_error_event = std::make_shared<KEvent>(); // never signaled
    std::shared_ptr<KEvent> resolution_event = std::make_shared<KEvent>(); // never signaled: no docking
    u64 next_layer_id = 1;

    AppletState() {
        message_event->name = "applet message";
        suspended_tick_event->name = "suspended tick";
        gpu_error_event->name = "gpu error detected";
        // AM tells an application its focus state at launch; SDK titles wait for
        // this message in nn::oe::InitializeApplet (sdk+0x3394d0).
        Post(kMessageFocusStateChanged);
    }

    void Post(u32 msg) {
        messages.push_back(msg);
        message_event->signaled = true;
    }
};

AppletState& State() {
    static AppletState s;
    return s;
}

/// A sub-object libnx opens at startup but this program never calls.
std::shared_ptr<Service> Empty(const char* name) { return std::make_shared<Service>(name); }

class CommonStateGetter : public Service {
public:
    CommonStateGetter() : Service("am:ICommonStateGetter") {
        Register(0, "GetEventHandle", [](IpcContext& ctx) {
            // The guest closes its copy at exit; keep ours alive by handing out a new handle each time.
            ctx.out_copy.push_back(g_handles.Add(State().message_event));
        });
        Register(1, "ReceiveMessage", [](IpcContext& ctx) {
            auto& s = State();
            if (s.messages.empty()) {
                ctx.result = kResultNoMessage;
                return;
            }
            ctx.Push<u32>(s.messages.front());
            s.messages.pop_front();
            s.message_event->signaled = !s.messages.empty();
        });
        Register(5, "GetOperationMode", [](IpcContext& ctx) { ctx.Push<u8>(ReportDocked() ? 1 : 0); }); // handheld / docked
        Register(6, "GetPerformanceMode", [](IpcContext& ctx) { ctx.Push<u32>(ReportDocked() ? 1 : 0); }); // normal / boost
        Register(9, "GetCurrentFocusState", [](IpcContext& ctx) { ctx.Push<u8>(1); }); // InFocus
        Register(60, "GetDefaultDisplayResolution", [](IpcContext& ctx) { // of the operation mode
            ctx.Push<s32>(ReportDocked() ? 1920 : 1280);
            ctx.Push<s32>(ReportDocked() ? 1080 : 720);
        });
        Register(61, "GetDefaultDisplayResolutionChangeEvent", [](IpcContext& ctx) {
            ctx.out_copy.push_back(g_handles.Add(State().resolution_event));
        });
        Stub(66, "SetCpuBoostMode");
        Stub(900, "SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled"); // 11.0.0+
    }
};

class SelfController : public Service {
public:
    SelfController() : Service("am:ISelfController") {
        Register(0, "Exit", [](IpcContext&) { Log("am: application requested exit"); Exit(0); });
        // The runtime never asks the game to exit (no HOME menu), so the lock holds nothing back. Brothership
        // locks around writing a save.
        Stub(1, "LockExit");
        Stub(2, "UnlockExit");
        Stub(10, "SetScreenShotPermission");
        Stub(11, "SetOperationModeChangedNotification");
        Stub(12, "SetPerformanceModeChangedNotification");
        Stub(13, "SetFocusHandlingMode");
        Stub(14, "SetRestartMessageEnabled");
        Stub(16, "SetOutOfFocusSuspendingEnabled");
        Register(40, "CreateManagedDisplayLayer", [](IpcContext& ctx) { ctx.Push<u64>(State().next_layer_id++); });
        // { u64 layer id; u64 recording layer id }. The recording layer (for
        // capture) is not modelled; 0, as drippu does.
        Register(44, "CreateManagedDisplaySeparableLayer", [](IpcContext& ctx) {
            ctx.Push<u64>(State().next_layer_id++);
            ctx.Push<u64>(0);
        });
        Stub(62, "SetIdleTimeDetectionExtension");
        Stub(68, "SetAutoSleepDisabled");
        Register(91, "GetAccumulatedSuspendedTickChangedEvent", [](IpcContext& ctx) {
            ctx.out_copy.push_back(g_handles.Add(State().suspended_tick_event));
        });
    }
};

class WindowController : public Service {
public:
    WindowController() : Service("am:IWindowController") {
        Register(1, "GetAppletResourceUserId", [](IpcContext& ctx) { ctx.Push<u64>(kAppletResourceUserId); });
        Stub(10, "AcquireForegroundRights");
    }
};

/// am IStorage: a byte buffer, read and written through IStorageAccessor.
class StorageAccessor : public Service {
public:
    explicit StorageAccessor(std::shared_ptr<std::vector<u8>> data) : Service("am:IStorageAccessor"), data_(data) {
        Register(0, "GetSize", [this](IpcContext& ctx) { ctx.Push<s64>(static_cast<s64>(data_->size())); });
        Register(11, "Read", [this](IpcContext& ctx) {
            const u64 offset = ctx.Pop<u64>();
            const IpcBuffer out = ctx.Out(0);
            if (offset > data_->size()) {
                ctx.result = kResultOutOfBounds;
                return;
            }
            ctx.WriteOut(0, data_->data() + offset, std::min<u64>(out.size, data_->size() - offset));
        });
    }
    static constexpr u32 kResultOutOfBounds = 0x1F680; // am module 128, description 503

private:
    std::shared_ptr<std::vector<u8>> data_;
};

class AmStorage : public Service {
public:
    explicit AmStorage(std::vector<u8> data)
        : Service("am:IStorage"), data_(std::make_shared<std::vector<u8>>(std::move(data))) {
        Register(0, "Open", [this](IpcContext& ctx) { ctx.PushObject(std::make_shared<StorageAccessor>(data_)); });
    }

private:
    std::shared_ptr<std::vector<u8>> data_;
};

class ApplicationFunctions : public Service {
public:
    ApplicationFunctions() : Service("am:IApplicationFunctions") {
        Register(1, "PopLaunchParameter", [](IpcContext& ctx) {
            const u32 kind = ctx.Pop<u32>();
            if (kind != 2) { // 1, the application's user channel: empty on a normal launch
                ctx.result = kResultNoMessage;
                return;
            }
            // AccountPreselectedUser: magic, is_selected, uid, reserved (0x88 bytes).
            std::vector<u8> data(0x88);
            const u32 magic = 0xC79497CA;
            std::memcpy(data.data(), &magic, 4);
            data[4] = 1;
            std::memcpy(data.data() + 8, &kAccountUser, sizeof kAccountUser);
            ctx.PushObject(std::make_shared<AmStorage>(std::move(data)));
        });
        Register(20, "EnsureSaveData", [](IpcContext& ctx) { // (uid) -> bytes still required
            Log("am: EnsureSaveData: the save data exists");
            ctx.Push<s64>(0);
        });
        // Saves are host folders without a size limit, so growing one always succeeds and its size is ample.
        // Brothership sends 25 only for an existing save when its config SaveGameSizeMaxMB is not 0;
        // not seen in play so far.
        Register(25, "ExtendSaveData", [](IpcContext& ctx) { // (type, uid, normal size, journal size) -> bytes still required
            const u8 type = ctx.Pop<u8>();
            ctx.Pop<u64>(), ctx.Pop<u64>(); // UserId (Pop aligns)
            const u64 normal = ctx.Pop<u64>(), journal = ctx.Pop<u64>();
            Log("am: ExtendSaveData(type %u, %llu + %llu bytes): granted", type, (unsigned long long)normal,
                (unsigned long long)journal);
            ctx.Push<s64>(0);
        });
        Register(26, "GetSaveDataSize", [](IpcContext& ctx) { // (type, uid) -> normal size, journal size
            constexpr u64 kSize = 256ULL << 20;
            ctx.Push<u64>(kSize);
            ctx.Push<u64>(kSize);
        });
        Register(21, "GetDesiredLanguage", [](IpcContext& ctx) {
            // LanguageCode: the language tag as 8 bytes. The system picks one the game supports, so the game takes
            // the answer as is (RT_LANGUAGE; the dump holds the assets of all 14).
            const char* lang = options::Get("RT_LANGUAGE");
            if (!lang) lang = "en-US";
            u64 code = 0;
            std::memcpy(&code, lang, std::min<size_t>(std::strlen(lang), 8));
            ctx.Push<u64>(code);
        });
        Register(22, "SetTerminateResult", [](IpcContext& ctx) {
            // The application reports why it is about to abort.
            Log("am: SetTerminateResult(0x%x)", ctx.Pop<u32>());
        });
        Register(40, "NotifyRunning", [](IpcContext& ctx) { ctx.Push<u8>(0); });
        Register(130, "GetGpuErrorDetectedSystemEvent", [](IpcContext& ctx) { // 8.0.0+
            ctx.out_copy.push_back(g_handles.Add(State().gpu_error_event));
        });
    }
};

class ApplicationProxy : public Service {
public:
    ApplicationProxy() : Service("am:IApplicationProxy") {
        Register(0, "GetCommonStateGetter", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<CommonStateGetter>()); });
        Register(1, "GetSelfController", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<SelfController>()); });
        Register(2, "GetWindowController", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<WindowController>()); });
        Register(3, "GetAudioController", [](IpcContext& ctx) { ctx.PushObject(Empty("am:IAudioController")); });
        Register(4, "GetDisplayController", [](IpcContext& ctx) { ctx.PushObject(Empty("am:IDisplayController")); });
        Register(11, "GetLibraryAppletCreator", [](IpcContext& ctx) { ctx.PushObject(Empty("am:ILibraryAppletCreator")); });
        Register(20, "GetApplicationFunctions", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<ApplicationFunctions>()); });
        Register(1000, "GetDebugFunctions", [](IpcContext& ctx) { ctx.PushObject(Empty("am:IDebugFunctions")); });
    }
};

class AppletOE : public Service {
public:
    AppletOE() : Service("appletOE") {
        Register(0, "OpenApplicationProxy", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<ApplicationProxy>()); });
    }
};

class ApmSession : public Service {
public:
    ApmSession() : Service("apm:ISession") {
        Stub(0, "SetPerformanceConfiguration");
        Register(1, "GetPerformanceConfiguration", [](IpcContext& ctx) { ctx.Push<u32>(0x00010000); });
    }
};

class Apm : public Service {
public:
    Apm() : Service("apm") {
        Register(0, "OpenSession", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<ApmSession>()); });
        Register(1, "GetPerformanceMode", [](IpcContext& ctx) { ctx.Push<u32>(ReportDocked() ? 1 : 0); });
    }
};

} // namespace

bool ReportDocked() {
    static const bool docked = [] {
        const char* e = std::getenv("RT_DOCKED");
        const bool on = e && *e && *e != '0';
        if (on) Log("am: reporting docked mode (RT_DOCKED)");
        return on;
    }();
    return docked;
}

namespace {

const bool registered = [] {
    RegisterServiceFactory("appletOE", [] { return std::make_shared<AppletOE>(); });
    RegisterServiceFactory("apm", [] { return std::make_shared<Apm>(); });
    return true;
}();

} // namespace

void AppletPostMessage(u32 msg) { State().Post(msg); }

} // namespace rt
