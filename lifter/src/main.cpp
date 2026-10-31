// nxrecomp: lifts a Switch module (ELF / NRO / uncompressed NSO) to C, or
// several NSOs of one program (statically linked first).
//
// Usage: nxrecomp <input>... -o <out_dir> [--validate] [--no-address-seeds] [--no-export-seeds]
//                 [--extra-seeds <file>] [--no-trace-pc | --trace-pc=func] [--no-fpx] [--no-fp-helpers] [--no-simd] [--no-direct-mem | --inline-mem]
//                 [--base <hex>] [--link-only] [--patch <file.pchtxt | file.ips | dir>]...
//
// With several inputs, give them in load order (rtld main subsdk0.. sdk): they
// are linked at --base (default 0x8000000, the runtime's image base), and
// <out_dir> gets link.txt (the module layout) and slots.txt (every GOT slot the
// link bound, "<slot> <target>" in program-relative hex: the runtime redirects
// the slots of functions it overrides). --link-only stops there, without
// discovering or emitting code.
// --extra-seeds reads one hex offset per line (e.g. the runtime's gaps file);
// for a linked program the offsets are program-relative.
// --patch applies exefs mods (IPSwitch .pchtxt, IPS / IPS32 .ips; a directory: every such file in it) to the
// module whose build id they name, before linking and discovery (patch.h); <out_dir>/patches.txt lists each
// write. Repeatable; files for other game versions are skipped.

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "analysis.h"
#include "emit.h"
#include "image.h"
#include "link.h"
#include "patch.h"

using namespace nxrecomp;

namespace {

double Since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

void PrintModule(const std::string& label, const Image& img, const Analysis& a, double secs) {
    std::printf("%s: %s, .text 0x%llx-0x%llx, %zu relocations, %zu FDEs, %zu functions, "
                "%zu jump tables (%zu unresolved) [%.1fs]\n",
                label.c_str(), img.format.c_str(), (unsigned long long)img.text_begin,
                (unsigned long long)img.text_end, img.relocations.size(), img.fde_starts.size(),
                a.functions.size(), a.jump_tables.size(), a.unresolved_jump_tables, secs);
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> inputs;
    std::string out_dir, extra;
    std::vector<std::string> patch_paths;
    bool validate = false, link_only = false;
    u64 load_base = 0x8000000;
    DiscoveryOptions dopts;
    EmitOptions eopts;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--validate") {
            validate = true;
        } else if (a == "--no-export-seeds") {
            dopts.use_export_seeds = false;
        } else if (a == "--no-address-seeds") {
            dopts.use_address_seeds = false;
        } else if (a == "--extra-seeds" && i + 1 < argc) {
            extra = argv[++i];
        } else if (a == "--no-trace-pc") {
            eopts.trace = EmitOptions::Trace::kOff;
        } else if (a == "--trace-pc=func") {
            eopts.trace = EmitOptions::Trace::kFunc;
        } else if (a == "--no-fpx") {
            eopts.fpx = false;
        } else if (a == "--fp-helpers") { // the default (kept so older commands still work)
            eopts.fp_helpers = true;
        } else if (a == "--no-fp-helpers") {
            eopts.fp_helpers = false;
        } else if (a == "--no-simd") {
            eopts.simd = false;
        } else if (a == "--inline-mem") { // the last memory flag wins
            eopts.inline_mem = true;
            eopts.direct_mem = false;
        } else if (a == "--direct-mem") { // the default (kept so older commands still work)
            eopts.direct_mem = true;
            eopts.inline_mem = false;
        } else if (a == "--no-direct-mem") {
            eopts.direct_mem = false;
        } else if (a == "--base" && i + 1 < argc) {
            load_base = std::stoull(argv[++i], nullptr, 16);
        } else if (a == "--patch" && i + 1 < argc) {
            patch_paths.push_back(argv[++i]);
        } else if (a == "--link-only") {
            link_only = true;
        } else if (a[0] != '-') {
            inputs.push_back(a);
        } else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (inputs.empty() || (out_dir.empty() && !validate)) {
        std::fprintf(stderr, "usage: %s <input>... -o <out_dir> [--validate] [--no-address-seeds] [--no-export-seeds] "
                             "[--extra-seeds <file>] [--no-trace-pc | --trace-pc=func] [--no-fpx] [--no-fp-helpers] [--no-simd] [--no-direct-mem | --inline-mem] [--base <hex>] [--link-only] [--patch <file | dir>]...\n",
                     argv[0]);
        return 2;
    }
    try {
        std::vector<u64> extra_seeds;
        if (!extra.empty()) {
            std::ifstream f(extra);
            std::string line;
            while (std::getline(f, line)) {
                if (!line.empty()) {
                    extra_seeds.push_back(std::stoull(line, nullptr, 16));
                }
            }
        }
        if (!out_dir.empty()) {
            std::filesystem::create_directories(out_dir);
            eopts.out_dir = out_dir;
        }

        std::vector<PatchFile> patches;
        for (const auto& p : patch_paths) {
            auto more = LoadPatches(p);
            patches.insert(patches.end(), more.begin(), more.end());
        }
        // Applied to the loaded modules (before relocation, as Atmosphère's loader does).
        const auto apply_patches = [&](std::vector<Image>& modules) {
            if (patches.empty()) return;
            const std::string report = ApplyPatches(patches, modules);
            std::fputs(report.c_str(), stdout);
            if (!out_dir.empty()) std::ofstream(out_dir + "/patches.txt") << report;
        };

        Image img;
        Analysis a;
        if (inputs.size() == 1) {
            const auto t = std::chrono::steady_clock::now();
            std::vector<Image> one{LoadImage(inputs[0])};
            apply_patches(one);
            img = std::move(one[0]);
            dopts.extra_seeds = extra_seeds;
            a = Discover(img, dopts);
            PrintModule(inputs[0], img, a, Since(t));
            if (validate) {
                std::fputs(ValidateAgainstSymbols(img, a).c_str(), stdout);
            }
        } else {
            std::vector<Image> modules;
            for (const auto& in : inputs) {
                modules.push_back(LoadImage(in));
            }
            apply_patches(modules);
            LinkReport report;
            auto t = std::chrono::steady_clock::now();
            img = Link(modules, load_base, report);
            std::fputs(report.Text().c_str(), stdout);
            std::printf("linked at 0x%llx: 0x%zx bytes [%.1fs]\n", (unsigned long long)load_base,
                        img.mem.size(), Since(t));
            if (!out_dir.empty()) {
                std::ofstream(out_dir + "/link.txt") << report.Text();
                std::ofstream slots(out_dir + "/slots.txt");
                for (const auto& [slot, target] : img.bound_slots) {
                    slots << std::hex << slot << ' ' << target << '\n';
                }
            }
            if (!report.strong_unresolved.empty()) {
                throw std::runtime_error("unresolved strong symbols");
            }
            if (link_only) {
                std::printf("%zu GOT slots statically bound\n", img.bound_slots.size());
                return 0;
            }
            std::vector<Analysis> parts;
            for (size_t i = 0; i < modules.size(); ++i) {
                const LinkedModule& lm = report.modules[i];
                dopts.extra_seeds.clear();
                dopts.linked_seeds.clear();
                for (u64 s : extra_seeds) {
                    if (s >= lm.offset && s < lm.offset + lm.size) {
                        dopts.extra_seeds.push_back(s - lm.offset);
                    }
                }
                for (const auto& [slot, target] : img.bound_slots) {
                    if (target >= lm.offset && target < lm.offset + lm.size) {
                        dopts.linked_seeds.push_back(target - lm.offset);
                    }
                }
                t = std::chrono::steady_clock::now();
                parts.push_back(Discover(modules[i], dopts));
                PrintModule(modules[i].name, modules[i], parts.back(), Since(t));
                if (validate) {
                    std::fputs(ValidateAgainstSymbols(modules[i], parts.back()).c_str(), stdout);
                }
            }
            a = MergeAnalyses(parts, report.modules);
            std::printf("%zu GOT slots statically bound\n", img.bound_slots.size());
            eopts.load_base = load_base;
        }
        if (!out_dir.empty()) {
            const auto t = std::chrono::steady_clock::now();
            const EmitStats s = Emit(img, a, eopts);
            std::printf("emitted %zu functions, %zu instructions (%zu unhandled) in %zu files; "
                        "%zu calls, %zu tail calls, %zu jump tables, %zu indirect, %zu svc, "
                        "%zu plt direct [%.1fs]\n",
                        s.functions, s.instructions, s.unhandled, s.files, s.direct_calls,
                        s.tail_calls, s.jump_tables, s.indirect_jumps, s.svcs, s.plt_direct,
                        Since(t));
            if (eopts.fp_helpers) {
                std::printf("fp helpers: %zu calls to %zu helpers, %zu FP/SIMD bodies left inline\n", s.fp_calls,
                            s.fp_helpers, s.fp_inline);
            }
            if (eopts.simd) {
                std::printf("simd: %zu sites (%zu encodings) as host vector code; FP fast path at %zu sites\n",
                            s.simd_sites, s.simd_encodings, s.fpx_sites);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
