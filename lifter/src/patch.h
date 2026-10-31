// Code and data patches (exefs mods) applied to the modules before the lift, as Atmosphère's loader applies
// them to an NSO before rtld relocates it: IPSwitch text patches (.pchtxt) and binary IPS / IPS32 (.ips).
// A patch names the module it is for by build id (`@nsobid-<hex>` in a .pchtxt, the file name of an .ips);
// it is applied only to a module whose build id starts with those bytes.
#pragma once

#include <string>
#include <vector>

#include "image.h"

namespace nxrecomp {

struct PatchWrite {
    u64 ips_offset; // as an IPS file counts it: from the start of the NSO file (header 0x100) or the NRO
    std::vector<u8> bytes;
};

struct PatchFile {
    std::string source; // the --patch argument it came from (the file itself or a directory)
    std::string path;
    std::vector<u8> build_id;
    std::vector<PatchWrite> writes; // enabled sections only, in file order (a later write wins)
};

/// Reads one .pchtxt / .ips file, or every such file under a directory (recursively, sorted by path).
/// Throws std::runtime_error on a malformed file.
std::vector<PatchFile> LoadPatches(const std::string& path);

/// Applies each patch to the module whose build id matches. Files that match no module are skipped (a mod ships
/// one per game version), but every source must apply at least one. Every write must lie in a module's
/// initialised bytes. Returns a report: a summary line per applied file, then one line per write.
std::string ApplyPatches(const std::vector<PatchFile>& patches, std::vector<Image>& modules);

} // namespace nxrecomp
