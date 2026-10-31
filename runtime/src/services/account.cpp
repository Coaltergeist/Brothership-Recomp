// acc:u0 (accounts, application side): one local user, "Player".
#include "../ipc.h"
#include "../services.h"

namespace rt {
namespace {

using Uid = AccountUid;
constexpr Uid kUser = kAccountUser;

/// ProfileBase: uid, last edit time, nickname (0x20 bytes, NUL-padded).
struct ProfileBase {
    Uid uid;
    u64 last_edit_timestamp;
    char nickname[0x20];
};
static_assert(sizeof(ProfileBase) == 0x38);

ProfileBase Base() {
    ProfileBase b{};
    b.uid = kUser;
    std::snprintf(b.nickname, sizeof b.nickname, "Player");
    return b;
}

/// Writes the user list (one user, the rest zero) to out buffer 0.
void ListUsers(IpcContext& ctx) {
    const IpcBuffer out = ctx.Out(0);
    std::vector<u8> list(out.size);
    if (list.size() >= sizeof(Uid)) {
        std::memcpy(list.data(), &kUser, sizeof kUser);
    }
    ctx.WriteOut(0, list.data(), list.size());
}

class Profile : public Service {
public:
    Profile() : Service("acc:IProfile") {
        Register(0, "Get", [](IpcContext& ctx) {
            const ProfileBase b = Base();
            ctx.PushBytes(&b, sizeof b);
            std::vector<u8> user_data(ctx.Out(0).size); // UserData: avatar settings, all zero
            ctx.WriteOut(0, user_data.data(), user_data.size());
        });
        Register(1, "GetBase", [](IpcContext& ctx) {
            const ProfileBase b = Base();
            ctx.PushBytes(&b, sizeof b);
        });
        Register(10, "GetImageSize", [](IpcContext& ctx) { ctx.Push<u32>(0); }); // no avatar image
    }
};

/// An async request that has already finished successfully (as drippu's
/// EnsureTokenIdCacheAsyncInterface: complete from the start).
class AsyncContext : public Service {
public:
    AsyncContext() : Service("acc:IAsyncContext") {
        event_->signaled = true;
        Register(0, "GetSystemEvent", [this](IpcContext& ctx) { ctx.out_copy.push_back(g_handles.Add(event_)); });
        Stub(1, "Cancel");
        Register(2, "HasDone", [](IpcContext& ctx) { ctx.Push<u8>(1); });
        Stub(3, "GetResult");
        event_->name = "account async";
    }

private:
    std::shared_ptr<KEvent> event_ = std::make_shared<KEvent>();
};

/// The user's network service account, as drippu reports it: available, with
/// an account id derived from the uid and empty token caches.
class ManagerForApplication : public Service {
public:
    ManagerForApplication() : Service("acc:IManagerForApplication") {
        Stub(0, "CheckAvailability");
        Register(1, "GetAccountId", [](IpcContext& ctx) { ctx.Push(AccountId()); });
        Register(2, "EnsureIdTokenCacheAsync", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<AsyncContext>()); });
        auto load_token = [](IpcContext& ctx) {
            const std::vector<u8> token(std::min<u64>(ctx.Out(0).size, 0x100));
            ctx.WriteOut(0, token.data(), token.size());
            ctx.Push<u32>(static_cast<u32>(token.size()));
        };
        Register(3, "LoadIdTokenCacheDeprecated", load_token);
        Register(4, "LoadIdTokenCache", load_token);
        auto resource_cache = [](IpcContext& ctx) {
            for (size_t i = 0; i < 2; ++i) {
                const std::vector<u8> zero(ctx.Out(i).size);
                if (!zero.empty()) ctx.WriteOut(i, zero.data(), zero.size());
            }
            ctx.Push(AccountId());
        };
        Register(130, "GetNintendoAccountUserResourceCacheForApplication", resource_cache);
        Register(136, "GetNintendoAccountUserResourceCache", resource_cache);
        Stub(160, "StoreOpenContext");
    }

private:
    static u64 AccountId() {
        u64 words[2];
        std::memcpy(words, &kUser, sizeof words);
        return words[0] ^ words[1];
    }
};

class AccountServiceForApplication : public Service {
public:
    AccountServiceForApplication() : Service("acc:u0") {
        Register(0, "GetUserCount", [](IpcContext& ctx) { ctx.Push<s32>(1); });
        Register(1, "GetUserExistence", [](IpcContext& ctx) {
            const Uid u = ctx.Pop<Uid>();
            ctx.Push<u8>(u.lo == kUser.lo && u.hi == kUser.hi);
        });
        Register(2, "ListAllUsers", ListUsers);
        Register(3, "ListOpenUsers", ListUsers);
        Register(4, "GetLastOpenedUser", [](IpcContext& ctx) { ctx.Push(kUser); });
        Register(5, "GetProfile", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<Profile>()); });
        Register(50, "IsUserRegistrationRequestPermitted", [](IpcContext& ctx) { ctx.Push<u8>(0); });
        Register(51, "TrySelectUserWithoutInteraction", [](IpcContext& ctx) { ctx.Push(kUser); });
        Stub(100, "InitializeApplicationInfoV0");
        Register(101, "GetBaasAccountManagerForApplication",
                 [](IpcContext& ctx) { ctx.PushObject(std::make_shared<ManagerForApplication>()); });
        Stub(140, "InitializeApplicationInfo");
        Register(150, "IsUserAccountSwitchLocked", [](IpcContext& ctx) { ctx.Push<u8>(0); });
    }
};

const bool registered = [] {
    RegisterServiceFactory("acc:u0", [] { return std::make_shared<AccountServiceForApplication>(); });
    return true;
}();

} // namespace
} // namespace rt
