#include "link.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace nxrecomp {

namespace {

constexpr u32 R_AARCH64_ABS64 = 257, R_AARCH64_GLOB_DAT = 1025, R_AARCH64_JUMP_SLOT = 1026,
              R_AARCH64_RELATIVE = 1027;
constexpr u8 STB_LOCAL = 0, STB_WEAK = 2;

u64 AlignUp(u64 v, u64 a) { return (v + a - 1) & ~(a - 1); }

std::string Hex(u64 v) {
    std::ostringstream o;
    o << "0x" << std::hex << v;
    return o.str();
}

} // namespace

Image Link(const std::vector<Image>& modules, u64 load_base, LinkReport& report) {
    Image prog;
    prog.format = "linked";
    prog.name = "program";

    // Layout: as the Horizon loader, consecutive page-aligned modules in order.
    u64 at = 0;
    for (const auto& m : modules) {
        const u64 size = AlignUp(m.mem.size(), 0x1000);
        auto rel = [&](u64 v) { return v ? at + v : 0; };
        report.modules.push_back({m.name, at, size, rel(m.init), rel(m.fini), rel(m.init_array),
                                  m.init_array_size, rel(m.fini_array), m.fini_array_size});
        prog.mem.resize(at + size, 0);
        std::memcpy(prog.mem.data() + at, m.mem.data(), m.mem.size());
        for (auto s : m.segments) {
            s.offset += at;
            prog.segments.push_back(s);
        }
        for (u64 f : m.fde_starts) {
            prog.fde_starts.push_back(at + f);
        }
        for (const auto& [off, name] : m.symbols) {
            prog.symbols.emplace(at + off, name);
        }
        at += size;
    }
    prog.entry = report.modules.front().offset + modules.front().entry;
    prog.text_begin = modules.front().text_begin + report.modules.front().offset;
    prog.text_end = report.modules.back().offset + modules.back().text_end;

    // Global scope: the first module (in load order) defining a name wins, so a
    // definition in main interposes on the same name in sdk (malloc, opus, ...).
    std::unordered_map<std::string, std::pair<size_t, u64>> global; // name -> (module, offset)
    for (size_t mi = 0; mi < modules.size(); ++mi) {
        for (const auto& s : modules[mi].dynsyms) {
            if (s.defined && s.binding != STB_LOCAL && !s.name.empty()) {
                global.try_emplace(s.name, mi, s.absolute ? s.value - load_base
                                                          : report.modules[mi].offset + s.value);
            }
        }
    }

    for (size_t mi = 0; mi < modules.size(); ++mi) {
        const Image& m = modules[mi];
        const u64 base = report.modules[mi].offset;
        for (const auto& r : m.relocations) {
            const u64 slot = base + r.offset;
            u64 value;
            if (r.type == R_AARCH64_RELATIVE) {
                value = load_base + base + static_cast<u64>(r.addend);
                ++report.relative;
            } else if (r.type == R_AARCH64_ABS64 || r.type == R_AARCH64_GLOB_DAT ||
                       r.type == R_AARCH64_JUMP_SLOT) {
                ++report.symbolic;
                if (r.symbol >= m.dynsyms.size()) {
                    throw std::runtime_error(m.name + ": relocation symbol index out of range");
                }
                const DynSymbol& s = m.dynsyms[r.symbol];
                u64 target = 0; // program-relative
                bool found = true;
                if (r.symbol == 0) {
                    found = false; // symbol 0: S = 0
                } else if (s.defined && (s.visibility != 0 || s.binding == STB_LOCAL)) {
                    target = s.absolute ? s.value - load_base : base + s.value;
                } else if (auto g = global.find(s.name); g != global.end()) {
                    target = g->second.second;
                    if (g->second.first != mi) {
                        s.defined ? ++report.interposed : ++report.bound_cross_module;
                    }
                } else if (s.binding == STB_WEAK) {
                    found = false;
                    report.weak_unresolved.push_back(m.name + ": " + s.name);
                } else {
                    found = false;
                    report.strong_unresolved.push_back(m.name + ": " + s.name);
                }
                value = found ? load_base + target + static_cast<u64>(r.addend)
                              : static_cast<u64>(r.addend);
                if (found && r.type != R_AARCH64_ABS64 && r.addend == 0) {
                    prog.bound_slots[slot] = target;
                }
            } else {
                throw std::runtime_error(m.name + ": unsupported relocation type " +
                                         std::to_string(r.type) + " at " + Hex(r.offset));
            }
            if (slot + 8 > prog.mem.size()) {
                throw std::runtime_error(m.name + ": relocation outside the module");
            }
            // JUMP_SLOTs stay as the module has them (the offset of PLT0): rtld does
            // lazy binding and refuses to start unless every slot still points there.
            // The emitter still uses the binding (bound_slots) for guarded direct calls.
            if (r.type != R_AARCH64_JUMP_SLOT) {
                std::memcpy(prog.mem.data() + slot, &value, 8);
            }
        }
    }
    // Deduplicate the unresolved lists (one entry per module and name).
    for (auto* v : {&report.weak_unresolved, &report.strong_unresolved}) {
        std::sort(v->begin(), v->end());
        v->erase(std::unique(v->begin(), v->end()), v->end());
    }
    return prog;
}

Analysis MergeAnalyses(const std::vector<Analysis>& parts, const std::vector<LinkedModule>& modules) {
    Analysis out;
    for (size_t i = 0; i < parts.size(); ++i) {
        const u64 base = modules[i].offset;
        for (const auto& [off, src] : parts[i].seeds) {
            out.seeds[base + off] = src;
        }
        for (auto f : parts[i].functions) {
            f.begin += base;
            f.end += base;
            out.functions.push_back(f);
        }
        for (const auto& [br, targets] : parts[i].jump_tables) {
            auto& t = out.jump_tables[base + br];
            for (u64 x : targets) {
                t.push_back(base + x);
            }
        }
        out.unresolved_jump_tables += parts[i].unresolved_jump_tables;
        for (u64 br : parts[i].unresolved_jump_table_sites) {
            out.unresolved_jump_table_sites.push_back(base + br);
        }
    }
    return out;
}

std::string LinkReport::Text() const {
    std::ostringstream o;
    o << "modules (program-relative):\n";
    for (const auto& m : modules) {
        o << "  " << m.name << " " << Hex(m.offset) << " size " << Hex(m.size) << " init "
          << Hex(m.init) << " init_array " << Hex(m.init_array) << " (" << m.init_array_size / 8
          << ") fini_array " << Hex(m.fini_array) << " (" << m.fini_array_size / 8 << ")\n";
    }
    o << "relocations: " << relative << " relative, " << symbolic << " symbolic ("
      << bound_cross_module << " imports bound to another module, " << interposed
      << " own definitions interposed by an earlier module)\n";
    o << "weak unresolved (bound to 0): " << weak_unresolved.size() << "\n";
    for (const auto& s : weak_unresolved) {
        o << "  " << s << "\n";
    }
    o << "strong unresolved: " << strong_unresolved.size() << "\n";
    for (const auto& s : strong_unresolved) {
        o << "  " << s << "\n";
    }
    return o.str();
}

} // namespace nxrecomp
