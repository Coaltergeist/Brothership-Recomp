// lm (log manager): the SDK's diagnostic log. nn::lm::Initialize (from
// nndiagStartup) opens it at startup and does not cope with its absence. Log
// packets are decoded and their text printed, so the SDK's own messages appear
// in the runtime log.
#include "../ipc.h"

namespace rt {
namespace {

// Packet: header, then chunks of (u8 key, ULEB128 size, data).
struct LogPacketHeader {
    u64 process_id, thread_id;
    u8 flags, padding, severity, verbosity;
    u32 payload_size;
};
static_assert(sizeof(LogPacketHeader) == 0x18);

enum LogKey : u8 { kTextLog = 2, kLineNumber = 3, kFileName = 4, kFunctionName = 5, kModuleName = 6 };

void PrintPacket(const std::vector<u8>& p) {
    if (p.size() < sizeof(LogPacketHeader)) {
        return;
    }
    LogPacketHeader h;
    std::memcpy(&h, p.data(), sizeof h);
    std::string text, where;
    for (size_t at = sizeof h; at < p.size();) {
        const u8 key = p[at++];
        u64 size = 0;
        for (int shift = 0; at < p.size(); shift += 7) {
            const u8 b = p[at++];
            size |= u64{b & 0x7Fu} << shift;
            if (!(b & 0x80)) {
                break;
            }
        }
        if (at + size > p.size()) {
            break;
        }
        const std::string value(reinterpret_cast<const char*>(p.data() + at), size);
        if (key == kTextLog) {
            text += value;
        } else if (key == kFunctionName) {
            where = value;
        }
        at += size;
    }
    while (!text.empty() && (text.back() == '\n' || text.back() == '\0')) {
        text.pop_back();
    }
    if (!text.empty()) {
        Log("lm   [sev %u]%s%s %s", h.severity, where.empty() ? "" : " ", where.c_str(), text.c_str());
    }
}

class Logger : public Service {
public:
    Logger() : Service("lm:ILogger") {
        Register(0, "Log", [](IpcContext& ctx) { PrintPacket(ctx.ReadIn(0)); });
        Stub(1, "SetDestination");
    }
};

class LogService : public Service {
public:
    LogService() : Service("lm") {
        Register(0, "OpenLogger", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<Logger>()); });
    }
};

const bool registered = [] {
    RegisterServiceFactory("lm", [] { return std::make_shared<LogService>(); });
    return true;
}();

} // namespace
} // namespace rt
