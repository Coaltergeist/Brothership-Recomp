// sm: (service manager) and set:sys (system settings).
#include "../ipc.h"
#include "../npdm.h"
#include "../threads.h"

namespace rt {
namespace {

constexpr u32 kResultSmNotRegistered = 0xE15; // sm module 21, description 7
constexpr u32 kResultSmNotAllowed = 0x1015;   // description 8: not in the NPDM's access list

std::string DecodeServiceName(u64 v) {
    std::string s;
    for (int i = 0; i < 8 && (v >> (8 * i)) & 0xFF; ++i) {
        s.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
    }
    return s;
}

class SmService : public Service {
public:
    SmService() : Service("sm:") {
        Stub(0, "RegisterClient");
        Register(1, "GetServiceHandle", [](IpcContext& ctx) {
            const std::string name = DecodeServiceName(ctx.Pop<u64>());
            // A TIPC reply keeps the command's output layout even on failure: the
            // SDK (nn::sm::GetServiceHandle) reads the result from word 4 and the
            // handle from word 3 without looking at the header, so an error reply
            // without the handle slot reads as success with handle 0.
            auto fail = [&](u32 result) {
                ctx.result = result;
                if (ctx.tipc) ctx.out_move.push_back(0);
            };
            if (!TitleNpdm().MayAccessService(name)) {
                Log("ipc  sm: GetServiceHandle(\"%s\"): not allowed by the NPDM", name.c_str());
                return fail(kResultSmNotAllowed);
            }
            auto service = MakeService(name);
            if (!service) {
                Log("ipc  sm: GetServiceHandle(\"%s\"): no HLE implementation", name.c_str());
                return fail(kResultSmNotRegistered);
            }
            Log("ipc  sm: GetServiceHandle(\"%s\") from %s", name.c_str(), Describe(CurrentContext()->x[30]).c_str());
            ctx.PushObject(service);
        });
        Stub(4, "DetachClient");
    }
};

/// Reported firmware: what libnx and the SDK use to pick command variants.
struct FirmwareVersion {
    u8 major, minor, micro, padding1, revision_major, revision_minor, padding2, padding3;
    char platform[0x20], version_hash[0x40], display_version[0x18], display_title[0x80];
};
static_assert(sizeof(FirmwareVersion) == 0x100);

class SetSysService : public Service {
public:
    SetSysService() : Service("set:sys") {
        auto get_version = [](IpcContext& ctx) {
            FirmwareVersion v{};
            v.major = 18;
            v.minor = 1;
            v.micro = 0;
            std::snprintf(v.platform, sizeof v.platform, "NX");
            std::snprintf(v.display_version, sizeof v.display_version, "18.1.0");
            std::snprintf(v.display_title, sizeof v.display_title, "NintendoSDK Firmware for NX 18.1.0");
            ctx.WriteOut(0, &v, sizeof v);
        };
        Register(3, "GetFirmwareVersion", get_version);
        Register(4, "GetFirmwareVersion2", get_version);
    }
};

const bool registered = [] {
    RegisterServiceFactory("sm:", [] { return std::make_shared<SmService>(); });
    RegisterServiceFactory("set:sys", [] { return std::make_shared<SetSysService>(); });
    return true;
}();

} // namespace
} // namespace rt
