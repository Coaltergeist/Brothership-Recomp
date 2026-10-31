// Static linking of a multi-module program (rtld, main, subsdkN, sdk) into one
// image at a fixed load address, doing at build time what Horizon's rtld does
// at run time.
#pragma once

#include <string>
#include <vector>

#include "analysis.h"
#include "image.h"

namespace nxrecomp {

struct LinkedModule {
    std::string name;
    u64 offset, size; // program-relative placement
    u64 init, fini, init_array, init_array_size, fini_array, fini_array_size; // program-relative
};

struct LinkReport {
    std::vector<LinkedModule> modules;
    size_t relative = 0, symbolic = 0, bound_cross_module = 0, interposed = 0;
    std::vector<std::string> weak_unresolved;   // "module: name", bound to 0
    std::vector<std::string> strong_unresolved; // an error
    std::string Text() const;
};

/// Places the modules consecutively (page-aligned, in the given order, which is
/// also the symbol lookup order) at `load_base`, binds every symbolic
/// relocation and applies all relocations. The result's addresses are
/// program-relative; pointers stored in it are absolute (load_base + offset).
Image Link(const std::vector<Image>& modules, u64 load_base, LinkReport& report);

/// Shifts each module's analysis by its placement and concatenates them.
Analysis MergeAnalyses(const std::vector<Analysis>& parts, const std::vector<LinkedModule>& modules);

} // namespace nxrecomp
