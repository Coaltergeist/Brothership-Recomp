// Shim for drippu's recompiler store-guard hook (core/arm/recomp). drippu's
// recompiled code checks stores into device-mapped pages; our
// lifted code has no such guards yet, so nothing is watched.
#pragma once
#include <cstddef>
#include "common/common_types.h"

namespace Core::RecompGuardGen {

inline bool Watching() {
    return false;
}

template <typename Entries>
inline void OnDeviceMap(const Entries&, VAddr, std::size_t) {}

} // namespace Core::RecompGuardGen
