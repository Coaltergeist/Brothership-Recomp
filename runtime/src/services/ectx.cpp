// ectx:aw (error context, 11.0.0+): the SDK opens it at startup
// (nn::err::InitializeContextWriterApi) when the firmware is new enough.
#include "../ipc.h"

namespace rt {
namespace {

class ErrorContextWriter : public Service {
public:
    ErrorContextWriter() : Service("ectx:aw") {}
};

const bool registered = [] {
    RegisterServiceFactory("ectx:aw", [] { return std::make_shared<ErrorContextWriter>(); });
    return true;
}();

} // namespace
} // namespace rt
