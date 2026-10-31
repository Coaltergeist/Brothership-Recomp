// nifm:u (network interface manager), bsd:u (sockets) and ssl: an offline console.
// nifm never accepts a connection request; bsd accepts registration and fails
// socket calls with ENETDOWN. UE4 initialises both at startup and goes on
// offline. Command ids are switchbrew's; ones not listed here stop the
// runtime with their id.
#include "../ipc.h"

namespace rt {
namespace {

constexpr s32 kENetDown = 50; // BSD errno, as Horizon's socket service reports

std::shared_ptr<KEvent> NeverSignalled(const char* name) {
    auto e = std::make_shared<KEvent>();
    e->name = name;
    return e;
}

class Request : public Service {
public:
    Request() : Service("nifm:IRequest") {
        // RequestState: 0 Invalid, 1 Free, 2 OnHold, 3 Accepted, 4 Blocking. On
        // an offline console a submitted request falls back to Free with an
        // error result: it is never accepted.
        Register(0, "GetRequestState", [](IpcContext& ctx) { ctx.Push<s32>(1); });
        Register(1, "GetResult", [](IpcContext& ctx) { ctx.result = kResultNoInternetConnection; });
        Register(2, "GetSystemEventReadableHandles", [](IpcContext& ctx) {
            ctx.out_copy.push_back(g_handles.Add(NeverSignalled("nifm request event")));
            ctx.out_copy.push_back(g_handles.Add(NeverSignalled("nifm request event 2")));
        });
        Stub(3, "Cancel");
        Stub(4, "Submit");
        Stub(6, "SetRequirementPreset");
        Stub(11, "SetConnectionConfirmationOption");
        Stub(21, "SetKeptInSleep");
        Stub(22, "RegisterSocketDescriptor");
        Stub(23, "UnregisterSocketDescriptor");
    }

    // nifm module 110, "network not available" *(assumption: the exact
    // description the SDK expects is not established)*.
    static constexpr u32 kResultNoInternetConnection = (110 << 0) | (3400 << 9);
};

class GeneralService : public Service {
public:
    GeneralService() : Service("nifm:IGeneralService") {
        Register(1, "GetClientId", [](IpcContext& ctx) {
            const u32 id = 1;
            ctx.WriteOut(0, &id, sizeof id);
        });
        Register(4, "CreateRequest", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<Request>()); });
        Register(12, "GetCurrentIpAddress", [](IpcContext& ctx) { ctx.result = Request::kResultNoInternetConnection; });
        Register(18, "GetInternetConnectionStatus", [](IpcContext& ctx) {
            ctx.result = Request::kResultNoInternetConnection;
        });
        Register(21, "IsAnyInternetRequestAccepted", [](IpcContext& ctx) { ctx.Push<u8>(0); });
    }
};

class NifmStatic : public Service {
public:
    NifmStatic() : Service("nifm:u") {
        auto create = [](IpcContext& ctx) { ctx.PushObject(std::make_shared<GeneralService>()); };
        Register(4, "CreateGeneralServiceOld", create);
        Register(5, "CreateGeneralService", create);
    }
};

/// Socket calls answer (ret, errno).
void Fail(IpcContext& ctx) {
    ctx.Push<s32>(-1);
    ctx.Push<s32>(kENetDown);
}

class BsdClient : public Service {
public:
    explicit BsdClient(const char* n) : Service(n) {
        Register(0, "RegisterClient", [](IpcContext& ctx) { ctx.Push<s32>(0); });
        Stub(1, "StartMonitoring");
        Register(2, "Socket", Fail);
        Register(3, "SocketExempt", Fail);
        Register(14, "Connect", Fail);
        Register(26, "Close", [](IpcContext& ctx) {
            ctx.Push<s32>(0);
            ctx.Push<s32>(0);
        });
    }
};

/// ssl: libcurl in UE4's HTTP module initialises it at startup. Contexts are
/// created, but with no network nothing ever connects through them.
class SslContext : public Service {
public:
    SslContext() : Service("ssl:ISslContext") {}
};

class SslService : public Service {
public:
    SslService() : Service("ssl") {
        Register(0, "CreateContext", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<SslContext>()); });
        Stub(5, "SetInterfaceVersion");
    }
};

const bool registered = [] {
    RegisterServiceFactory("ssl", [] { return std::make_shared<SslService>(); });
    RegisterServiceFactory("nifm:u", [] { return std::make_shared<NifmStatic>(); });
    for (const char* n : {"bsd:u", "bsd:s"}) {
        RegisterServiceFactory(n, [n] { return std::make_shared<BsdClient>(n); });
    }
    return true;
}();

} // namespace
} // namespace rt
