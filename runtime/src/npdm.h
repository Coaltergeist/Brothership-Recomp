// The title's NPDM (main.npdm from the ExeFS): what Horizon's loader and sm
// know about the process. Read from RT_NPDM; without it (homebrew), the
// program id is 0 and every service is allowed.
#pragma once
#include <string>
#include <vector>
#include "rt.h"

namespace rt {

struct Npdm {
    bool loaded = false;
    u64 program_id = 0;
    std::vector<std::string> services; // service access control: names the process may open
    bool MayAccessService(const std::string& name) const;
};

const Npdm& TitleNpdm();

} // namespace rt
