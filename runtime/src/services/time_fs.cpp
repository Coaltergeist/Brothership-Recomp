// time:* (clocks and time zones).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>

#include "../ipc.h"

namespace rt {
namespace {

class SystemClock : public Service {
public:
    explicit SystemClock(const char* n) : Service(n) {
        Register(0, "GetCurrentTime", [](IpcContext& ctx) { // the host's clock (Unix time), as the shared memory's
            using namespace std::chrono;
            ctx.Push<s64>(duration_cast<seconds>(system_clock::now().time_since_epoch()).count());
        });
    }
};

class SteadyClock : public Service {
public:
    SteadyClock() : Service("time:ISteadyClock") {}
};

/// Days since 1970-01-01 of a civil date (proleptic Gregorian).
s64 DaysFromCivil(s64 y, unsigned m, unsigned d) {
    y -= m <= 2;
    const s64 era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    return era * 146097 + static_cast<s64>(yoe * 365 + yoe / 4 - yoe / 100 + doy) - 719468;
}

/// CalendarTime (8 bytes) + CalendarAdditionalInfo (24 bytes) for a POSIX time, in the host's local time zone:
/// the layer's time-zone rule stands for the host's zone (GetDeviceLocationName, LoadTimeZoneRule), so both
/// conversions use it — a save made at 15:00 shows 15:00. Zone name: the host's abbreviation (8 characters).
void ToCalendar(IpcContext& ctx, s64 posix) {
    const std::time_t t = static_cast<std::time_t>(posix);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    const s64 as_utc = DaysFromCivil(tm.tm_year + 1900, static_cast<unsigned>(tm.tm_mon + 1), static_cast<unsigned>(tm.tm_mday)) * 86400 +
                       tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec;
    const s32 gmt_offset = static_cast<s32>(as_utc - posix);
    char zone[16] = {};
    std::strftime(zone, sizeof zone, "%Z", &tm);
    u8 out[32] = {};
    const int16_t year = static_cast<int16_t>(tm.tm_year + 1900);
    std::memcpy(out, &year, 2);
    out[2] = static_cast<u8>(tm.tm_mon + 1);
    out[3] = static_cast<u8>(tm.tm_mday);
    out[4] = static_cast<u8>(tm.tm_hour);
    out[5] = static_cast<u8>(tm.tm_min);
    out[6] = static_cast<u8>(tm.tm_sec);
    const u32 wday = static_cast<u32>(tm.tm_wday), yday = static_cast<u32>(tm.tm_yday);
    std::memcpy(out + 8, &wday, 4);
    std::memcpy(out + 12, &yday, 4);
    std::memcpy(out + 16, zone, std::min<size_t>(std::strlen(zone), 7)); // timezone name (8)
    const u32 dst = tm.tm_isdst > 0;
    std::memcpy(out + 24, &dst, 4);
    std::memcpy(out + 28, &gmt_offset, 4);
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 10) {
        Log("time: %lld (Unix time) is %04d-%02d-%02d %02d:%02d:%02d %s (UTC%+d s)", (long long)posix, year, out[2], out[3],
            out[4], out[5], out[6], zone, gmt_offset);
    }
    ctx.PushBytes(out, sizeof out);
}

class TimeZoneService : public Service {
public:
    TimeZoneService() : Service("time:ITimeZoneService") {
        Register(0, "GetDeviceLocationName", [](IpcContext& ctx) {
            char name[0x24] = "UTC"; // LocationName
            ctx.PushBytes(name, sizeof name);
        });
        // A UTC rule. Conversions are computed here (always UTC), so the guest
        // only passes the rule back; its contents matter only if the guest
        // parses it: timecnt 0, typecnt 1, everything else zero.
        Register(4, "LoadTimeZoneRule", [](IpcContext& ctx) {
            std::vector<u8> rule(0x4000);
            const s32 typecnt = 1;
            std::memcpy(rule.data() + 4, &typecnt, 4);
            ctx.WriteOut(0, rule.data(), rule.size());
        });
        Register(100, "ToCalendarTime", [](IpcContext& ctx) { ToCalendar(ctx, ctx.Pop<s64>()); });
        Register(101, "ToCalendarTimeWithMyRule", [](IpcContext& ctx) { ToCalendar(ctx, ctx.Pop<s64>()); });
    }
};

/// time:u/a/s. With shared memory (6.0+) the SDK and libnx read clocks from it directly (Brothership never calls
/// GetCurrentTime): a system clock is its context's offset plus the steady clock, the steady clock is the guest
/// tick counter in seconds plus the steady entry's base (nanoseconds; zero here, as the continuous-adjustment
/// entry), and the tick counter is the host's steady clock (rt.cpp ReadCntpct). So the local and network contexts
/// (layout: drippu psc/time/shared_memory.h) get offset = Unix time - steady seconds, the host's clock. The steady
/// entry and both contexts carry one non-zero clock source id: the SDK's CalculateSystemClockCurrentTime (sdk
/// 0x3a5ccc) fails (0xcc74) for a zero id or one that differs from the steady clock's, and the user clock then
/// reads 0 - saves dated 1970-01-01 + minutes even after the offset was set (2026-11-04).
class TimeService : public Service {
public:
    explicit TimeService(const char* n) : Service(n) {
        Register(0, "GetStandardUserSystemClock", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<SystemClock>("time:ISystemClock(user)")); });
        Register(1, "GetStandardNetworkSystemClock", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<SystemClock>("time:ISystemClock(network)")); });
        Register(2, "GetStandardSteadyClock", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<SteadyClock>()); });
        Register(3, "GetTimeZoneService", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<TimeZoneService>()); });
        Register(4, "GetStandardLocalSystemClock", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<SystemClock>("time:ISystemClock(local)")); });
        Register(20, "GetSharedMemoryNativeHandle", [](IpcContext& ctx) {
            static auto shmem = [] {
                auto s = std::make_shared<KSharedMemory>(0x1000);
                s->name = "time";
                using namespace std::chrono;
                const s64 steady = duration_cast<seconds>(steady_clock::now().time_since_epoch()).count();
                const s64 unix_now = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
                struct Context { // SystemClockContext: offset, SteadyClockTimePoint {time point, clock source id}
                    s64 offset, time_point;
                    u8 source_id[16];
                } context{unix_now - steady, steady, {'b', 'r', 'o', 't', 'h', 'e', 'r', 's', 'h', 'i', 'p', '-', 'r', 't', 1, 0}};
                static_assert(sizeof(Context) == 0x20);
                for (const u64 at : {u64{0x38}, u64{0x80}}) { // LockFreeAtomicType: counter 0 -> value [0] at +8
                    std::memcpy(s->data.data() + at + 8, &context, sizeof context);
                    std::memcpy(s->data.data() + at + 8 + sizeof context, &context, sizeof context);
                }
                for (const u64 at : {u64{8}, u64{8 + 0x18}}) { // the steady entries {s64 base, source id}: the id
                    std::memcpy(s->data.data() + at + 8, context.source_id, sizeof context.source_id);
                }
                Log("time: system clocks at %lld (Unix time), from the host", (long long)unix_now);
                return s;
            }();
            ctx.out_copy.push_back(g_handles.Add(shmem));
        });
    }
};

const bool registered = [] {
    for (const char* n : {"time:u", "time:a", "time:s"}) {
        RegisterServiceFactory(n, [n] { return std::make_shared<TimeService>(n); });
    }
    return true;
}();

} // namespace
} // namespace rt
