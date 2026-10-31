#include "npdm.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

namespace rt {

bool Npdm::MayAccessService(const std::string& name) const {
    if (!loaded) return true;
    for (const auto& s : services) {
        // A trailing '*' grants a whole prefix.
        if (s == name || (!s.empty() && s.back() == '*' && name.compare(0, s.size() - 1, s, 0, s.size() - 1) == 0)) {
            return true;
        }
    }
    return false;
}

namespace {

Npdm Load() {
    Npdm n;
    const char* path = std::getenv("RT_NPDM");
    if (!path) return n;
    std::ifstream f(path, std::ios::binary);
    const std::vector<u8> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u32_at = [&](size_t o) {
        u32 v = 0;
        if (o + 4 <= d.size()) std::memcpy(&v, d.data() + o, 4);
        return v;
    };
    if (d.size() < 0x80 || std::memcmp(d.data(), "META", 4) != 0) Fatal("RT_NPDM: %s is not an NPDM", path);
    const u32 aci = u32_at(0x70); // META: ACI0 offset
    if (aci + 0x40 > d.size() || std::memcmp(d.data() + aci, "ACI0", 4) != 0) Fatal("RT_NPDM: no ACI0");
    std::memcpy(&n.program_id, d.data() + aci + 0x10, 8);
    const u32 sac = aci + u32_at(aci + 0x28), sac_size = u32_at(aci + 0x2C);
    // Service access control entries: a control byte (bit 7: host/register,
    // bits 0-2: name length - 1), then the name.
    for (u32 i = 0; i < sac_size && sac + i < d.size();) {
        const u8 ctrl = d[sac + i];
        const u32 len = (ctrl & 7) + 1;
        if (!(ctrl & 0x80)) n.services.emplace_back(reinterpret_cast<const char*>(d.data() + sac + i + 1), len);
        i += 1 + len;
    }
    n.loaded = true;
    Log("npdm program id %016llx, %zu services allowed", (unsigned long long)n.program_id, n.services.size());
    return n;
}

} // namespace

const Npdm& TitleNpdm() {
    static const Npdm n = Load();
    return n;
}

} // namespace rt
