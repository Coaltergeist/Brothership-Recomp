#include "emit.h"

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <cctype>
#include <set>
#include <iterator>
#include <map>
#include <stdexcept>

#include "core/recompiler/arm64_to_c.h"
#include "fastmem.h"
#include "simd.h"

namespace nxrecomp {

namespace {

namespace dr = suyu::recomp;

std::string Hex(u64 v) {
    char b[32];
    std::snprintf(b, sizeof b, "%" PRIx64, v);
    return b;
}

std::string FnName(u64 off) { return "f_" + Hex(off); }
std::string Abs(u64 off) { return "(g_module_base+0x" + Hex(off) + "ULL)"; }

// Runtime contract shared by every generated file (see runtime/src/rt.h).
constexpr const char* kPrelude = R"C(#pragma once
#include "recomp_runtime.h"

/* Supplied by the runtime. */
void rt_svc(GuestContext* c, uint32_t imm);
void rt_jump(GuestContext* c, uint64_t target); /* call whatever lives at target */
void rt_break(GuestContext* c, uint32_t imm, uint64_t pc);
extern _Thread_local uint64_t rt_trace_pc;
#define RT_PC(off) (rt_trace_pc = (off))

typedef struct RtFunc { uint64_t offset; void (*fn)(GuestContext*); } RtFunc;
extern const RtFunc rt_funcs[];
extern const unsigned long rt_func_count;
/* Image segments; perm bits: 1 read, 2 write, 4 execute. */
typedef struct RtSegment { uint64_t offset, size; uint32_t perm; } RtSegment;
extern const RtSegment rt_segments[];
extern const unsigned long rt_segment_count;
/* rt_load_base: 0 if the image relocates itself, else the address it was linked at. */
extern const uint64_t rt_image_size, rt_entry, rt_load_base;
)C";

class FunctionEmitter {
public:
    FunctionEmitter(const Image& img, const Analysis& a, const EmitOptions& o, EmitStats& s)
        : img_(img), a_(a), opts_(o), stats_(s) {}

    std::string Emit(const Function& f) {
        out_.clear();
        f_ = &f;
        std::set<u64> labels;
        for (u64 pc = f.begin; pc < f.end; pc += 4) {
            u64 t;
            if (DirectTarget(img_.Word(pc), pc, t) && (img_.Word(pc) & 0xFC000000) != 0x94000000 &&
                t >= f.begin && t < f.end) {
                labels.insert(t);
            }
            if (auto jt = a_.jump_tables.find(pc); jt != a_.jump_tables.end()) {
                labels.insert(jt->second.begin(), jt->second.end());
            }
        }
        auto sym = img_.symbols.find(f.begin);
        if (sym != img_.symbols.end()) {
            out_ += "/* " + sym->second + " */\n";
        }
        out_ += "void " + FnName(f.begin) + "(GuestContext* c) {\n";
        if (opts_.trace == EmitOptions::Trace::kFunc) {
            Line("RT_PC(0x" + Hex(f.begin) + "ULL);");
        }
        for (u64 pc = f.begin; pc < f.end; pc += 4) {
            if (labels.count(pc)) {
                out_ += "L_" + Hex(pc) + ":;\n";
            }
            Instruction(pc, img_.Word(pc));
            ++stats_.instructions;
        }
        // Falling off the end of the range continues in the next function.
        const u32 last = img_.Word(f.end - 4);
        if (!IsUnconditionalTransfer(last)) {
            if (const Function* next = a_.Find(f.end); next && next->begin == f.end) {
                Line(FnName(f.end) + "(c); return;");
                ++stats_.tail_calls;
            } else {
                Line("rt_break(c, 0xFFFFu, " + Abs(f.end) + "); return; /* runs into padding */");
            }
        }
        out_ += "}\n\n";
        ++stats_.functions;
        return out_;
    }

private:
    void Line(const std::string& s) { out_ += "    " + s + "\n"; }

    // Branch to a module offset: a goto inside this function, otherwise a tail call.
    std::string Goto(u64 t) {
        if (t >= f_->begin && t < f_->end) {
            return "goto L_" + Hex(t) + ";";
        }
        ++stats_.tail_calls;
        if (const Function* g = a_.Find(t); g && g->begin == t) {
            return "{ " + FnName(t) + "(c); return; }";
        }
        return "{ rt_jump(c, " + Abs(t) + "); return; }";
    }

    // `br x17` ending `adrp x16, P; ldr x17, [x16, #o]; add x16, x16, #o` (a PLT
    // entry) with a statically bound slot P+o and a lifted target.
    bool PltTarget(u64 pc, u64& target) const {
        if (img_.Word(pc) != 0xD61F0220 || pc < f_->begin + 12) {
            return false;
        }
        const u32 adrp = img_.Word(pc - 12), ldr = img_.Word(pc - 8), add = img_.Word(pc - 4);
        if ((adrp & 0x9F00001F) != 0x90000010 || (ldr & 0xFFC003FF) != 0xF9400211 ||
            (add & 0xFFC003FF) != 0x91000210 || ((add >> 10) & 0xFFF) != ((ldr >> 10) & 0xFFF) * 8) {
            return false;
        }
        const s64 imm = (static_cast<s64>(static_cast<s32>(((adrp >> 5) & 0x7FFFF) << 13)) >> 13) << 2 |
                        ((adrp >> 29) & 3);
        const u64 slot = ((pc - 12) & ~0xFFFULL) + static_cast<u64>(imm << 12) + ((ldr >> 10) & 0xFFF) * 8;
        auto b = img_.bound_slots.find(slot);
        if (b == img_.bound_slots.end()) {
            return false;
        }
        const Function* g = a_.Find(b->second);
        if (!g || g->begin != b->second) {
            return false;
        }
        target = b->second;
        return true;
    }

    // After a call returns, continue only if the callee returned here; otherwise
    // the guest switched return address (longjmp, hand-set x30), so unwind.
    std::string ResumeCheck(u64 next) {
        std::string s = "if (c->pc != " + Abs(next) + ") return;";
        if (opts_.trace == EmitOptions::Trace::kFunc) {
            s += " RT_PC(0x" + Hex(next - 4) + "ULL);";
        }
        return s;
    }

    void Instruction(u64 pc, u32 i) {
        const u64 next = pc + 4;
        char cmt[48];
        std::snprintf(cmt, sizeof cmt, "/* %" PRIx64 ": %08x */", pc, i);
        if (opts_.trace == EmitOptions::Trace::kInsn) {
            Line(std::string("RT_PC(0x") + Hex(pc) + "ULL); " + cmt);
        } else {
            Line(cmt);
        }
        u64 t;
        if ((i & 0xFC000000) == 0x14000000 && DirectTarget(i, pc, t)) { // B
            Line(Goto(t));
        } else if ((i & 0xFC000000) == 0x94000000 && DirectTarget(i, pc, t)) { // BL
            ++stats_.direct_calls;
            const Function* g = a_.Find(t);
            Line("c->x[30] = " + Abs(next) + ";");
            Line(g && g->begin == t ? FnName(t) + "(c);" : "rt_jump(c, " + Abs(t) + ");");
            Line(ResumeCheck(next));
        } else if ((i & 0xFF000010) == 0x54000000 && DirectTarget(i, pc, t)) { // B.cond
            Line("if " + dr::Cond(i & 15) + " " + Goto(t));
        } else if ((i & 0x7E000000) == 0x34000000 && DirectTarget(i, pc, t)) { // CBZ/CBNZ
            const std::string v = (i >> 31) ? dr::Xz(i & 31) : dr::Wz(i & 31);
            Line("if ((" + v + ")" + ((i >> 24) & 1 ? " != " : " == ") + "0) " + Goto(t));
        } else if ((i & 0x7E000000) == 0x36000000 && DirectTarget(i, pc, t)) { // TBZ/TBNZ
            const u32 bit = ((i >> 31) << 5) | ((i >> 19) & 31);
            Line("if (((" + dr::Xz(i & 31) + " >> " + std::to_string(bit) + ") & 1)" +
                 ((i >> 24) & 1 ? " != " : " == ") + "0) " + Goto(t));
        } else if ((i & 0xFFFFFC1F) == 0xD65F0000) { // RET
            Line("c->pc = " + dr::Xz((i >> 5) & 31) + "; return;");
        } else if ((i & 0xFFFFFC1F) == 0xD61F0000) { // BR
            if (auto jt = a_.jump_tables.find(pc); jt != a_.jump_tables.end()) {
                ++stats_.jump_tables;
                std::set<u64> cases(jt->second.begin(), jt->second.end());
                std::string sw = "switch (" + dr::Xz((i >> 5) & 31) + " - g_module_base) {";
                for (u64 t : cases) {
                    if (t >= f_->begin && t < f_->end) { // a loosely bounded table may reach into padding
                        sw += " case 0x" + Hex(t) + "ULL: goto L_" + Hex(t) + ";";
                    }
                }
                Line(sw + " }");
                Line("rt_jump(c, " + dr::Xz((i >> 5) & 31) + "); return; /* not a known case */");
            } else if (u64 t; PltTarget(pc, t)) {
                // A PLT stub whose GOT slot the static linker bound: call the target
                // directly while the slot still holds it (rtld's lazy binding or the
                // game could change it).
                ++stats_.plt_direct;
                Line("if (c->x[17] == " + Abs(t) + ") { " + FnName(t) + "(c); return; }");
                Line("rt_jump(c, c->x[17]); return;");
            } else {
                ++stats_.indirect_jumps;
                Line("rt_jump(c, " + dr::Xz((i >> 5) & 31) + "); return;");
            }
        } else if ((i & 0xFFFFFC1F) == 0xD63F0000) { // BLR
            ++stats_.indirect_jumps;
            Line("{ uint64_t _t = " + dr::Xz((i >> 5) & 31) + "; c->x[30] = " + Abs(next) +
                 "; rt_jump(c, _t); }");
            Line(ResumeCheck(next));
        } else if ((i & 0xFFE0001F) == 0xD4000001) { // SVC
            ++stats_.svcs;
            Line("c->pc = " + Abs(next) + "; rt_svc(c, 0x" + Hex((i >> 5) & 0xFFFF) + "u);");
        } else if (i == 0xE7FFDEFE) { // clang's trap word (after noreturn calls)
            Line("rt_break(c, 0xDEFEu, " + Abs(pc) + "); return;");
        } else if ((i & 0xFFE0001F) == 0xD4200000) { // BRK
            Line("rt_break(c, 0x" + Hex((i >> 5) & 0xFFFF) + "u, " + Abs(pc) + "); return;");
        } else if ((i & 0xFFFFF0FF) == 0xD50330DF) { // ISB
            Line("recomp_barrier();");
        } else if ((i & 0xFFFFFFE0) == 0xD50B7520) { // IC IVAU: no runtime code generation
            Line("/* ic ivau */");
        } else if ((i & 0xFFFFFFE0) == 0xD53B00E0) { // MRS Xt, DCZID_EL0
            // 64-byte DC ZVA blocks, as on the Cortex-A57 and in drippu's DC ZVA.
            if ((i & 31) != 31) {
                Line("c->x[" + std::to_string(i & 31) + "] = 4;");
            }
        } else {
            std::string body;
            if (opts_.simd && ((i >> 25) & 7) == 7 && simd::Emit(i, body)) { // FP/SIMD data processing
                simd_words_.insert(i);
                ++stats_.simd_sites;
                out_ += body;
                return;
            }
            bool miss = false;
            if (!dr::Translate(i, pc, body, &miss)) {
                throw std::runtime_error("Translate ended a block at 0x" + Hex(pc) +
                                         " for an instruction the emitter does not intercept");
            }
            stats_.unhandled += miss;
            if (opts_.fp_helpers && ((i >> 25) & 7) == 7) { // FP/SIMD data processing (op0 = x111)
                if (!miss && SelfContained(body, pc)) {
                    auto [it, fresh] = fp_helpers_.emplace(i, body);
                    if (fresh || it->second == body) {
                        // FADD/FSUB/FMUL/FADDP: native on all lanes under FPX's conditions, the helper otherwise
                        if (opts_.simd && opts_.fpx && simd::EmitFp(i, FpName(i) + "(c);", out_)) {
                            ++stats_.fpx_sites;
                        } else {
                            Line(FpName(i) + "(c);");
                        }
                        ++stats_.fp_calls;
                        return;
                    }
                }
                ++stats_.fp_inline;
            }
            out_ += body;
        }
    }

    // A body that depends only on the instruction word: no control flow out of it, no labels, nothing
    // derived from its address (so one copy serves every place the encoding occurs).
    static bool SelfContained(const std::string& body, u64 pc) {
        for (const char* s : {"goto", "return", "L_", "c->pc", "rt_", "g_module_base"}) {
            if (body.find(s) != std::string::npos) {
                return false;
            }
        }
        return body.find(Hex(pc)) == std::string::npos;
    }

public:
    static std::string FpName(u32 i) {
        char b[16];
        std::snprintf(b, sizeof b, "fp_%08x", i);
        return b;
    }
    const std::map<u32, std::string>& FpHelpers() const { return fp_helpers_; }
    size_t SimdEncodings() const { return simd_words_.size(); }

private:
    std::map<u32, std::string> fp_helpers_; // encoding -> drippu's translation
    std::set<u32> simd_words_;               // encodings emitted as host vector code

    const Image& img_;
    const Analysis& a_;
    const EmitOptions& opts_;
    EmitStats& stats_;
    const Function* f_ = nullptr;
    std::string out_;
};

// Leaves an identical file untouched, so a rebuild only recompiles what changed.
void WriteFile(const std::string& path, const std::string& data) {
    {
        std::ifstream old(path, std::ios::binary);
        if (old) {
            std::string have((std::istreambuf_iterator<char>(old)), {});
            if (have == data) {
                return;
            }
        }
    }
    std::ofstream f(path, std::ios::binary);
    if (!f || !f.write(data.data(), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("cannot write " + path);
    }
}

} // namespace

EmitStats Emit(const Image& img, const Analysis& a, const EmitOptions& opts) {
    dr::g_translate_all = true; // no JIT behind us
    dr::g_emit_fpx = opts.fpx;
    const std::string dir = opts.out_dir + "/";
    EmitStats stats;

    // The generated code calls fabs/fabsf, which drippu's header does not declare: GCC takes them as
    // builtins, clang 16+ (the Windows build) rejects the undeclared call. The include goes ahead of
    // drippu's header so drippu itself stays unmodified.
    WriteFile(dir + "recomp_runtime.h", std::string("#include <math.h>\n") + dr::RuntimeH());
    WriteFile(dir + "recomp_runtime.c", dr::RuntimeC());

    // The shared header holds only the runtime contract; each file declares the
    // functions it calls, so a change to the function list recompiles only the
    // files it touches.
    WriteFile(dir + "nx_mem.h", opts.direct_mem ? fastmem::DirectHeader() : fastmem::Header());
    WriteFile(dir + "nx_funcs.h",
              std::string(kPrelude) + (opts.inline_mem || opts.direct_mem ? "#include \"nx_mem.h\"\n" : ""));
    auto prototypes = [](const std::string& code) {
        // Every call is spelled "f_<hex>(c)" (functions) or "fp_<hex>(c)" (FP helpers).
        std::set<std::string> names;
        for (const std::string prefix : {"f_", "fp_"}) {
            for (size_t at = code.find(prefix); at != std::string::npos; at = code.find(prefix, at + prefix.size())) {
                size_t end = at + prefix.size();
                while (end < code.size() && std::isxdigit(static_cast<unsigned char>(code[end]))) {
                    ++end;
                }
                if (end > at + prefix.size() && code.compare(end, 3, "(c)") == 0 &&
                    (at == 0 || !std::isalnum(static_cast<unsigned char>(code[at - 1])))) {
                    names.insert(code.substr(at, end - at));
                }
            }
        }
        std::string out;
        for (const auto& n : names) {
            out += "void " + n + "(GuestContext* c);\n";
        }
        return out;
    };

    FunctionEmitter fe(img, a, opts, stats);
    std::string file;
    auto flush = [&] {
        if (file.empty()) {
            return;
        }
        char name[32];
        std::snprintf(name, sizeof name, "funcs_%04zu.c", stats.files++);
        // Only a file with vector code includes nx_simd.h, so output without any stays as it was.
        const char* simd = file.find(simd::kMarker) != std::string::npos ? "#include \"nx_simd.h\"\n" : "";
        WriteFile(dir + name, "#include \"nx_funcs.h\"\n" + std::string(simd) + "\n" + prototypes(file) + "\n" + file);
        file.clear();
    };
    for (const auto& f : a.functions) {
        file += fe.Emit(f);
        if (file.size() > opts.max_file_bytes) {
            flush();
        }
    }
    flush();

    // The FP helpers, sorted by encoding, in files of their own (funcs_fp_*.c: the build's funcs_*.c glob).
    size_t fp_files = 0;
    auto flush_fp = [&] {
        if (file.empty()) {
            return;
        }
        char name[32];
        std::snprintf(name, sizeof name, "funcs_fp_%04zu.c", fp_files++);
        WriteFile(dir + name, "#include \"nx_funcs.h\"\n\n" + file);
        file.clear();
    };
    for (const auto& [word, body] : fe.FpHelpers()) {
        file += "void " + FunctionEmitter::FpName(word) + "(GuestContext* c) {\n" + body + "}\n\n";
        if (file.size() > opts.max_file_bytes) {
            flush_fp();
        }
    }
    flush_fp();
    stats.fp_helpers = fe.FpHelpers().size();
    stats.simd_encodings = fe.SimdEncodings();
    if (stats.simd_sites) {
        WriteFile(dir + "nx_simd.h", simd::Header());
    }
    stats.files += fp_files;

    std::string table = "#include \"nx_funcs.h\"\n\n";
    for (const auto& f : a.functions) {
        table += "void " + FnName(f.begin) + "(GuestContext* c);\n";
    }
    table += "\nconst RtFunc rt_funcs[] = {\n";
    for (const auto& f : a.functions) {
        table += "    {0x" + Hex(f.begin) + "ULL, " + FnName(f.begin) + "},\n";
    }
    table += "};\nconst unsigned long rt_func_count = " + std::to_string(a.functions.size()) + ";\n";
    table += "const uint64_t rt_image_size = 0x" + Hex(img.mem.size()) + "ULL;\n";
    table += "const uint64_t rt_entry = 0x" + Hex(img.entry) + "ULL;\n";
    table += "const uint64_t rt_load_base = 0x" + Hex(opts.load_base) + "ULL;\n";
    table += "const RtSegment rt_segments[] = {\n";
    for (const auto& s : img.segments) {
        // A self-relocating image writes its relocations into rodata (libnx __nx_dynamic).
        const bool write = s.write || (!s.exec && opts.load_base == 0);
        const u32 perm = 1 | (write ? 2 : 0) | (s.exec ? 4 : 0);
        table += "    {0x" + Hex(s.offset) + "ULL, 0x" + Hex(s.mem_size) + "ULL, " +
                 std::to_string(perm) + "},\n";
    }
    table += "};\nconst unsigned long rt_segment_count = " + std::to_string(img.segments.size()) + ";\n";
    if (opts.direct_mem) { // the runtime checks it against its window (rt.cpp); absent: the checked helpers
        table += "const unsigned long long nx_mem_direct_base = 0x" + Hex(fastmem::kDirectBase) + "ULL;\n";
    }
    WriteFile(dir + "nx_table.c", table);

    WriteFile(dir + "image.bin", std::string(img.mem.begin(), img.mem.end()));

    std::string syms;
    for (const auto& [off, name] : img.symbols) {
        syms += Hex(off) + " " + name + "\n";
    }
    WriteFile(dir + "symbols.txt", syms);
    return stats;
}

} // namespace nxrecomp
