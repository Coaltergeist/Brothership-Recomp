#include "kernel.h"

#include <cstring>

#include "ipc.h"

namespace rt {

HandleTable g_handles;

void KSharedMemory::Write(u64 offset, const void* src, u64 size) {
    std::memcpy(data.data() + offset, src, size);
    for (u64 va : mappings) {
        std::memcpy(g_mem.Raw(va + offset), src, size);
    }
}

u32 HandleTable::Add(std::shared_ptr<KObject> obj) {
    // Handles are allocated round-robin and wrap (games that open and close
    // sessions every frame get there); skip any still open. Like Horizon's
    // (15-bit index, 15-bit counter) they stay below bit 30: the SDK keeps a
    // mutex owner's handle in a word whose bit 30 means "has waiters", and a
    // thread whose handle had it set failed nn::os's ownership check.
    u32 h;
    for (u32 tries = 0;; ++tries) {
        if (tries > kHandleSlots) Fatal("handle table full (%u handles open)", static_cast<unsigned>(objects_.size()));
        h = next_;
        next_ += 0x10000;
        if (next_ >= 0x4000'0000) next_ = kFirstHandle;
        if (!objects_.count(h)) break;
    }
    objects_[h] = std::move(obj);
    return h;
}

std::shared_ptr<KObject> HandleTable::Get(u32 handle) const {
    auto it = objects_.find(handle);
    return it == objects_.end() ? nullptr : it->second;
}

bool HandleTable::Close(u32 handle) { return objects_.erase(handle) != 0; }

std::string HandleTable::Describe(u64 handle) const {
    char b[96];
    if (handle == kPseudoCurrentThread) return "CurrentThread";
    if (handle == kPseudoCurrentProcess) return "CurrentProcess";
    if (handle == kMainThreadHandle) return "thread:main";
    auto obj = Get(static_cast<u32>(handle));
    if (!obj) {
        std::snprintf(b, sizeof b, "0x%llx(invalid)", (unsigned long long)handle);
        return b;
    }
    if (auto s = std::dynamic_pointer_cast<KSession>(obj); s && s->object) {
        return std::string("session:") + s->object->name + (s->domain ? "(domain)" : "");
    }
    return std::string(obj->Kind()) + ":" + obj->name;
}

u32 OpenSession(std::shared_ptr<Service> object) {
    auto s = std::make_shared<KSession>();
    s->object = std::move(object);
    return g_handles.Add(s);
}

} // namespace rt
