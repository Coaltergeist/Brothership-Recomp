// aoc:u (add-on content): the game checks for DLC at startup. None is installed.
#include "../ipc.h"

namespace rt {
namespace {

constexpr u64 kAddOnContentBaseId = 0x0100'6D00'17F7'B000; // Brothership's application id + 0x1000

class AddOnContentManager : public Service {
public:
    AddOnContentManager() : Service("aoc:u") {
        Register(2, "CountAddOnContent", [](IpcContext& ctx) { ctx.Push<u32>(0); });
        Register(3, "ListAddOnContent", [](IpcContext& ctx) { ctx.Push<u32>(0); });
        Register(5, "GetAddOnContentBaseId", [](IpcContext& ctx) { ctx.Push<u64>(kAddOnContentBaseId); });
        auto list_changed = [](IpcContext& ctx) {
            static auto ev = [] {
                auto e = std::make_shared<KEvent>(); // never signalled: the list never changes
                e->name = "aoc list changed";
                return e;
            }();
            ctx.out_copy.push_back(g_handles.Add(ev));
        };
        Register(8, "GetAddOnContentListChangedEvent", list_changed);
        Register(10, "GetAddOnContentListChangedEventWithProcessId", list_changed);
        Stub(50, "CheckAddOnContentMountStatus");
    }
};

const bool registered = [] {
    RegisterServiceFactory("aoc:u", [] { return std::make_shared<AddOnContentManager>(); });
    return true;
}();

} // namespace
} // namespace rt
