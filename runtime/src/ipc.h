// IPC: HIPC message parsing, CMIF/TIPC dispatch, and HLE service objects.
#pragma once

#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kernel.h"

namespace rt {

constexpr u32 kResultSuccess = 0;
constexpr u64 kAppletResourceUserId = 0x51; // same value as the process id

/// Queues an applet message (e.g. 4 = ExitRequest) for ICommonStateGetter.
void AppletPostMessage(u32 msg);

struct IpcBuffer {
    u64 address = 0, size = 0;
};

/// One request as a service handler sees it, plus the response it builds.
class IpcContext {
public:
    u32 command = 0;
    bool tipc = false;
    bool has_pid = false;
    u64 pid = 0;
    std::vector<u32> copy_handles, move_handles, in_objects;
    std::vector<IpcBuffer> x, a, b, w, c;
    std::vector<u8> data; // request payload after the CMIF header (TIPC: all raw data)

    template <typename T>
    T Pop() {
        T v{};
        read_ = (read_ + alignof(T) - 1) & ~(alignof(T) - 1);
        if (read_ + sizeof(T) <= data.size()) {
            std::memcpy(&v, data.data() + read_, sizeof(T));
        }
        read_ += sizeof(T);
        return v;
    }
    template <typename T>
    T At(size_t offset) const {
        T v{};
        if (offset + sizeof(T) <= data.size()) {
            std::memcpy(&v, data.data() + offset, sizeof(T));
        }
        return v;
    }

    // Response.
    u32 result = kResultSuccess;
    std::vector<u8> out;
    std::vector<u32> out_copy, out_move;
    std::vector<std::shared_ptr<Service>> out_objects;

    template <typename T>
    void Push(const T& v) {
        const size_t at = (out.size() + alignof(T) - 1) & ~(alignof(T) - 1);
        out.resize(at + sizeof(T));
        std::memcpy(out.data() + at, &v, sizeof(T));
    }
    void PushBytes(const void* p, size_t n) {
        const size_t at = out.size();
        out.resize(at + n);
        std::memcpy(out.data() + at, p, n);
    }
    void PushObject(std::shared_ptr<Service> s) { out_objects.push_back(std::move(s)); }

    /// The i-th in buffer (A, or the matching X for auto-select buffers).
    IpcBuffer In(size_t i) const;
    /// The i-th out buffer (B, or the matching C for auto-select buffers).
    IpcBuffer Out(size_t i) const;
    std::vector<u8> ReadIn(size_t i) const;
    std::string ReadInString(size_t i) const;
    void WriteOut(size_t i, const void* p, size_t n) const;

private:
    size_t read_ = 0;
};

/// An HLE service object. Commands are registered by id; anything else is fatal,
/// so an unimplemented command never passes silently.
class Service : public std::enable_shared_from_this<Service> {
public:
    explicit Service(std::string service_name) { name = std::move(service_name); }
    virtual ~Service() = default;

    using Handler = std::function<void(IpcContext&)>;
    void Handle(IpcContext& ctx);

    std::string name;

protected:
    void Register(u32 cmd, const char* cmd_name, Handler h);
    /// Succeeds with `out_bytes` zero bytes of output, logging it as a stub.
    void Stub(u32 cmd, const char* cmd_name, size_t out_bytes = 0);

private:
    struct Entry {
        std::string name;
        Handler handler;
        bool stub;
    };
    std::map<u32, Entry> commands_;
};

/// Creates the HLE service registered under `name` (as sm:GetServiceHandle sees it).
std::shared_ptr<Service> MakeService(const std::string& name);
using ServiceFactory = std::function<std::shared_ptr<Service>()>;
void RegisterServiceFactory(const std::string& name, ServiceFactory f);

/// svcSendSyncRequest: parses the TLS message, dispatches it, writes the reply.
u32 SendSyncRequest(GuestContext* c, u32 handle);

extern bool g_trace_ipc;

} // namespace rt
