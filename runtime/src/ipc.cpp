#include "ipc.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>

namespace rt {

bool g_trace_ipc = true; // and RT_TRACE_SVC (g_trace_svc, rt.cpp), checked where used

namespace {

/// Whether a call of `name`.`command` is logged: always, or with RT_TRACE_SVC=first its first 100 times and
/// whenever it fails.
bool TraceIpcCall(const std::string& name, u32 command, u32 result) {
    if (g_trace_all || result != 0) return true;
    static std::mutex mu;
    static std::map<std::pair<std::string, u32>, u32> seen;
    std::lock_guard lock(mu);
    return seen[{name, command}]++ < 100;
}

constexpr u32 kCmifInMagic = 0x49434653;  // "SFCI"
constexpr u32 kCmifOutMagic = 0x4F434653; // "SFCO"
constexpr u32 kTypeClose = 2, kTypeRequest = 4, kTypeControl = 5, kTypeRequestWithContext = 6,
              kTypeControlWithContext = 7, kTipcClose = 15, kTipcBase = 16;
constexpr u32 kResultInvalidHandle = 0xE401;      // kernel: InvalidHandle
constexpr u16 kPointerBufferSize = 0x8000;        // as drippu (KProcess default); fixed X buffers need one

std::map<std::string, ServiceFactory>& Factories() {
    static std::map<std::string, ServiceFactory> f;
    return f;
}

IpcBuffer StaticDescriptor(u32 d0, u32 d1) {
    return {d1 | (u64{(d0 >> 12) & 0xF} << 32) | (u64{(d0 >> 6) & 0x3F} << 36), d0 >> 16};
}

IpcBuffer BufferDescriptor(u32 w0, u32 w1, u32 w2) {
    return {w1 | (u64{w2 >> 28} << 32) | (u64{(w2 >> 2) & 0x3FFFFF} << 36),
            w0 | (u64{(w2 >> 24) & 0xF} << 32)};
}

// Writes the response for `ctx` into the TLS message buffer.
void WriteResponse(u64 tls, KSession& session, IpcContext& ctx, bool domain_reply) {
    u32 msg[64] = {};
    std::vector<u32> object_ids, moves;
    for (auto& obj : ctx.out_objects) {
        if (domain_reply) {
            const u32 id = session.domain->next_object_id++;
            session.domain->objects[id] = obj;
            object_ids.push_back(id);
        } else {
            moves.push_back(OpenSession(obj));
        }
    }
    moves.insert(moves.end(), ctx.out_move.begin(), ctx.out_move.end());
    const bool special = !ctx.out_copy.empty() || !moves.empty();
    u32 idx = 2;
    if (special) {
        msg[idx++] = (static_cast<u32>(ctx.out_copy.size()) << 1) | (static_cast<u32>(moves.size()) << 5);
        for (u32 h : ctx.out_copy) msg[idx++] = h;
        for (u32 h : moves) msg[idx++] = h;
    }
    const u32 data_begin = idx;
    auto put_bytes = [&](u32 byte_off, const void* p, size_t n) {
        if (byte_off + n > sizeof msg) {
            Fatal("IPC response does not fit the 0x100-byte message buffer");
        }
        std::memcpy(reinterpret_cast<u8*>(msg) + byte_off, p, n);
    };
    u32 end;
    if (ctx.tipc) {
        u32 pos = data_begin * 4;
        put_bytes(pos, &ctx.result, 4);
        pos += 4;
        if (ctx.result == kResultSuccess) {
            put_bytes(pos, ctx.out.data(), ctx.out.size());
            pos += static_cast<u32>(ctx.out.size());
        }
        end = pos;
    } else {
        u32 pos = (data_begin * 4 + 15) & ~15u;
        if (domain_reply) {
            const u32 dom[4] = {static_cast<u32>(object_ids.size()), 0, 0, 0};
            put_bytes(pos, dom, sizeof dom);
            pos += sizeof dom;
        }
        const u32 hdr[4] = {kCmifOutMagic, 0, ctx.result, 0};
        put_bytes(pos, hdr, sizeof hdr);
        pos += sizeof hdr;
        if (ctx.result == kResultSuccess) {
            put_bytes(pos, ctx.out.data(), ctx.out.size());
            pos += static_cast<u32>(ctx.out.size());
            for (u32 id : object_ids) {
                put_bytes(pos, &id, 4);
                pos += 4;
            }
        }
        end = pos;
    }
    const u32 data_words = (end - data_begin * 4 + 3) / 4;
    msg[0] = 0; // response type
    msg[1] = data_words | (special ? 0x8000'0000u : 0);
    g_mem.Write(tls, msg, sizeof msg);
}

void ControlRequest(KSession& session, IpcContext& ctx) {
    switch (ctx.command) {
    case 0: // ConvertCurrentObjectToDomain
        if (!session.domain) {
            session.domain = std::make_shared<KDomain>();
            session.domain->objects[session.domain->next_object_id] = session.object;
            ctx.Push<u32>(session.domain->next_object_id++);
        } else {
            ctx.Push<u32>(1);
        }
        break;
    case 2: // CloneCurrentObject
    case 4: { // CloneCurrentObjectEx
        // A clone of a domain session joins the same domain.
        const u32 h = OpenSession(session.object);
        g_handles.GetAs<KSession>(h)->domain = session.domain;
        ctx.out_move.push_back(h);
        break;
    }
    case 3: // QueryPointerBufferSize
        ctx.Push<u16>(kPointerBufferSize);
        break;
    default:
        Fatal("unimplemented CMIF control command %u on %s", ctx.command, session.object->name.c_str());
    }
    if (g_trace_ipc && g_trace_svc) {
        static const char* names[] = {"ConvertCurrentObjectToDomain", "CopyFromCurrentDomain",
                                      "CloneCurrentObject", "QueryPointerBufferSize",
                                      "CloneCurrentObjectEx"};
        Log("ipc  %s control.%s", session.object->name.c_str(), names[ctx.command]);
    }
}

} // namespace

// --- IpcContext ----------------------------------------------------------------

IpcBuffer IpcContext::In(size_t i) const {
    if (i < a.size() && a[i].size) return a[i];
    if (i < x.size()) return x[i];
    return i < a.size() ? a[i] : IpcBuffer{};
}

IpcBuffer IpcContext::Out(size_t i) const {
    if (i < b.size() && b[i].size) return b[i];
    if (i < c.size()) return c[i];
    return i < b.size() ? b[i] : IpcBuffer{};
}

std::vector<u8> IpcContext::ReadIn(size_t i) const {
    const IpcBuffer buf = In(i);
    std::vector<u8> v(buf.size);
    if (buf.size) {
        g_mem.Read(buf.address, v.data(), buf.size);
    }
    return v;
}

std::string IpcContext::ReadInString(size_t i) const {
    const std::vector<u8> v = ReadIn(i);
    return std::string(v.begin(), std::find(v.begin(), v.end(), 0));
}

void IpcContext::WriteOut(size_t i, const void* p, size_t n) const {
    const IpcBuffer buf = Out(i);
    if (n > buf.size) {
        Fatal("IPC out buffer %zu holds 0x%llx bytes, 0x%zx needed", i, (unsigned long long)buf.size, n);
    }
    g_mem.Write(buf.address, p, n);
}

// --- Service ------------------------------------------------------------------

void Service::Register(u32 cmd, const char* cmd_name, Handler h) {
    commands_[cmd] = {cmd_name, std::move(h), false};
}

void Service::Stub(u32 cmd, const char* cmd_name, size_t out_bytes) {
    commands_[cmd] = {cmd_name, [out_bytes](IpcContext& ctx) { ctx.out.resize(out_bytes); }, true};
}

void Service::Handle(IpcContext& ctx) {
    auto it = commands_.find(ctx.command);
    if (it == commands_.end()) {
        Fatal("unimplemented command %u on %s (%zu payload bytes, %zu/%zu/%zu/%zu/%zu x/a/b/w/c buffers)",
              ctx.command, name.c_str(), ctx.data.size(), ctx.x.size(), ctx.a.size(), ctx.b.size(),
              ctx.w.size(), ctx.c.size());
    }
    // a request taking over 50 ms is logged whatever the trace setting: the caller held the kernel lock for it
    // unless the handler released it (fs.cpp's SaveIo and IStorage reads), and every other thread needing it waited
    const auto t0 = std::chrono::steady_clock::now();
    it->second.handler(ctx);
    if (const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); ms > 50) {
        Log("ipc: slow request %s.%s(%u): %.1f ms (t=%.1f s)", name.c_str(), it->second.name.c_str(), ctx.command, ms,
            Uptime());
    }
    if (g_trace_ipc && g_trace_svc && TraceIpcCall(name, ctx.command, ctx.result)) {
        Log("ipc  %s.%s(%u)%s -> 0x%x", name.c_str(), it->second.name.c_str(), ctx.command,
            it->second.stub ? " [stub]" : "", ctx.result);
    }
}

void RegisterServiceFactory(const std::string& name, ServiceFactory f) { Factories()[name] = std::move(f); }

std::shared_ptr<Service> MakeService(const std::string& name) {
    auto it = Factories().find(name);
    return it == Factories().end() ? nullptr : it->second();
}

// --- SendSyncRequest ------------------------------------------------------------

u32 SendSyncRequest(GuestContext* c, u32 handle) {
    auto session = g_handles.GetAs<KSession>(handle);
    if (!session) {
        Log("ipc  SendSyncRequest on %s from %s", g_handles.Describe(handle).c_str(), Describe(c->x[30]).c_str());
        // Guest frame-pointer chain: [fp] = caller's fp, [fp + 8] = return address.
        u64 fp = c->x[29];
        for (int i = 0; i < 8 && fp && g_mem.Find(fp) && g_mem.Find(fp + 8); ++i) {
            u64 next, lr;
            g_mem.Read(fp, &next, 8);
            g_mem.Read(fp + 8, &lr, 8);
            Log("ipc    #%d %s", i, Describe(lr).c_str());
            if (next <= fp) break;
            fp = next;
        }
        return kResultInvalidHandle;
    }
    const u64 tls = c->tpidrro_el0;
    u32 msg[64];
    g_mem.Read(tls, msg, sizeof msg);

    IpcContext ctx;
    const u32 type = msg[0] & 0xFFFF;
    const u32 num_x = (msg[0] >> 16) & 15, num_a = (msg[0] >> 20) & 15, num_b = (msg[0] >> 24) & 15,
              num_w = (msg[0] >> 28) & 15;
    const u32 num_data_words = msg[1] & 0x3FF, recv_mode = (msg[1] >> 10) & 15;
    u32 idx = 2;
    if (msg[1] >> 31) {
        const u32 sp = msg[idx++];
        const u32 copies = (sp >> 1) & 15, moves = (sp >> 5) & 15;
        if (sp & 1) {
            ctx.has_pid = true;
            ctx.pid = msg[idx] | (u64{msg[idx + 1]} << 32);
            idx += 2;
        }
        for (u32 n = 0; n < copies; ++n) ctx.copy_handles.push_back(msg[idx++]);
        for (u32 n = 0; n < moves; ++n) ctx.move_handles.push_back(msg[idx++]);
    }
    for (u32 n = 0; n < num_x; ++n, idx += 2) ctx.x.push_back(StaticDescriptor(msg[idx], msg[idx + 1]));
    for (u32 n = 0; n < num_a; ++n, idx += 3) ctx.a.push_back(BufferDescriptor(msg[idx], msg[idx + 1], msg[idx + 2]));
    for (u32 n = 0; n < num_b; ++n, idx += 3) ctx.b.push_back(BufferDescriptor(msg[idx], msg[idx + 1], msg[idx + 2]));
    for (u32 n = 0; n < num_w; ++n, idx += 3) ctx.w.push_back(BufferDescriptor(msg[idx], msg[idx + 1], msg[idx + 2]));
    const u32 data_idx = idx;
    const u32 num_recv = recv_mode == 2 ? 1 : recv_mode > 2 ? recv_mode - 2 : 0;
    for (u32 n = 0, r = data_idx + num_data_words; n < num_recv && r + 1 < 64; ++n, r += 2) {
        ctx.c.push_back({msg[r] | (u64{msg[r + 1] & 0xFFFF} << 32), msg[r + 1] >> 16});
    }
    const u8* bytes = reinterpret_cast<const u8*>(msg);
    const u32 data_end = std::min<u32>((data_idx + num_data_words) * 4, sizeof msg);

    if (type == kTypeClose || type == kTipcClose) {
        if (g_trace_ipc && g_trace_svc) Log("ipc  %s closed by client", session->object->name.c_str());
        return kResultSuccess;
    }
    if (type >= kTipcBase) {
        ctx.tipc = true;
        ctx.command = type - kTipcBase;
        ctx.data.assign(bytes + data_idx * 4, bytes + data_end);
        session->object->Handle(ctx);
        WriteResponse(tls, *session, ctx, false);
        return kResultSuccess;
    }
    if (type != kTypeRequest && type != kTypeControl && type != kTypeRequestWithContext &&
        type != kTypeControlWithContext) {
        Fatal("unsupported IPC message type %u on %s", type, session->object->name.c_str());
    }

    u32 pos = (data_idx * 4 + 15) & ~15u;
    const bool control = type == kTypeControl || type == kTypeControlWithContext;
    std::shared_ptr<Service> target = session->object;
    u32 payload_end = data_end;
    const bool domain_request = session->domain && !control;
    if (domain_request) {
        const u8 dtype = bytes[pos], num_objects = bytes[pos + 1];
        const u16 data_size = static_cast<u16>(bytes[pos + 2] | (bytes[pos + 3] << 8));
        u32 object_id;
        std::memcpy(&object_id, bytes + pos + 4, 4);
        auto it = session->domain->objects.find(object_id);
        if (it == session->domain->objects.end()) {
            Fatal("domain request to unknown object %u on %s", object_id, session->object->name.c_str());
        }
        target = it->second;
        if (dtype == 2) { // close object
            if (g_trace_ipc && g_trace_svc) Log("ipc  %s (domain object %u) closed", target->name.c_str(), object_id);
            const std::string closed = target->name;
            const auto t0 = std::chrono::steady_clock::now();
            session->domain->objects.erase(it);
            target.reset(); // the last reference, usually: the object's destructor runs here (e.g. a host file closed)
            if (const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ms > 50) {
                Log("ipc: slow close of %s: %.1f ms (t=%.1f s)", closed.c_str(), ms, Uptime());
            }
            WriteResponse(tls, *session, ctx, true);
            return kResultSuccess;
        }
        payload_end = pos + 16 + data_size;
        for (u32 n = 0; n < num_objects; ++n) {
            u32 id;
            std::memcpy(&id, bytes + payload_end + 4 * n, 4);
            ctx.in_objects.push_back(id);
        }
        pos += 16;
    }
    u32 hdr[4];
    std::memcpy(hdr, bytes + pos, sizeof hdr);
    if (hdr[0] != kCmifInMagic) {
        Fatal("CMIF request to %s has magic 0x%08x", target->name.c_str(), hdr[0]);
    }
    ctx.command = hdr[2];
    pos += 16;
    if (payload_end > pos) {
        ctx.data.assign(bytes + pos, bytes + std::min<u32>(payload_end, sizeof msg));
    }
    if (control) {
        ControlRequest(*session, ctx);
    } else {
        target->Handle(ctx);
    }
    WriteResponse(tls, *session, ctx, domain_request);
    return kResultSuccess;
}

} // namespace rt
