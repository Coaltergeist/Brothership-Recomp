// ldr:ro (dynamic module loading). Brothership initialises nn::ro and imports
// RegisterModuleInfo/UnloadModule, but ships no NROs and never loads one; a
// load would need code lifted ahead of time, so it stops the runtime.
#include "../ipc.h"

namespace rt {
namespace {

class RoService : public Service {
public:
    RoService() : Service("ldr:ro") {
        Register(0, "MapManualLoadModuleMemory", [](IpcContext&) {
            Fatal("the game loads an NRO at run time: it must be lifted ahead of time");
        });
        Stub(1, "UnmapManualLoadModuleMemory");
        Stub(2, "RegisterModuleInfo");   // NRR registration: nothing to verify against
        Stub(3, "UnregisterModuleInfo");
        Stub(4, "RegisterProcessHandle"); // nn::ro::Initialize
    }
};

const bool registered = [] {
    RegisterServiceFactory("ldr:ro", [] { return std::make_shared<RoService>(); });
    return true;
}();

} // namespace
} // namespace rt
