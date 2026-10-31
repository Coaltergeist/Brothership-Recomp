#include "analysis.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <cstdlib>

namespace nxrecomp {

namespace {

constexpr u32 R_AARCH64_ABS64 = 257, R_AARCH64_GLOB_DAT = 1025, R_AARCH64_JUMP_SLOT = 1026,
              R_AARCH64_RELATIVE = 1027;

bool IsBl(u32 i) { return (i & 0xFC000000) == 0x94000000; }

// Padding and traps that can sit between functions.
bool IsPadding(u32 i) {
    return i == 0 || i == 0xD503201F /* nop */ || i == 0xE7FFDEFE ||
           (i & 0xFFE0001F) == 0xD4200000 /* brk */ || (i & 0xFFFF0000) == 0 /* udf */;
}

void AddSeed(Analysis& a, const Image& img, u64 off, u32 source) {
    if (img.InText(off) && (off & 3) == 0) {
        a.seeds[off] |= source;
    }
}

bool IsBr(u32 i) { return (i & 0xFFFFFC1F) == 0xD61F0000; }

s64 AdrImm(u32 i) {
    return (static_cast<s64>(static_cast<s32>(((i >> 5) & 0x7FFFF) << 13)) >> 13) << 2 | ((i >> 29) & 3);
}
bool IsAdr(u32 i) { return (i & 0x9F000000) == 0x10000000; }
bool IsAdrp(u32 i) { return (i & 0x9F000000) == 0x90000000; }
bool IsAddImm64(u32 i) { return (i & 0xFFC00000) == 0x91000000; } // ADD Xd, Xn, #imm12 (no shift)

// Whether `i` may write general register `r` (0..30). Conservative: anything
// not known to leave `r` alone counts as a write, which only costs a missed
// jump table, never a wrong one.
bool MayWrite(u32 i, u32 r) {
    const u32 rd = i & 31, rn = (i >> 5) & 31, rt2 = (i >> 10) & 31;
    const u32 op0 = (i >> 25) & 0xF;
    if ((op0 & 0xE) == 0xA) { // branches, exceptions, system
        if ((i & 0xFC000000) == 0x94000000 || (i & 0xFFFFFC1F) == 0xD63F0000) {
            return r <= 18 || r == 30; // BL, BLR: the callee may clobber x0-x18
        }
        if ((i & 0xFFF00000) == 0xD5300000) {
            return rd == r; // MRS
        }
        return false;
    }
    if ((op0 & 0x5) == 0x4) { // loads and stores
        const bool vec = (i >> 26) & 1;
        const bool pair = ((i >> 27) & 7) == 5;
        const bool writeback = (pair && ((i >> 23) & 3) != 2 && ((i >> 23) & 3) != 0) ||
                               (!pair && ((i >> 24) & 3) == 0 && ((i >> 21) & 1) == 0 &&
                                ((i >> 10) & 3) != 0 && ((i >> 10) & 3) != 2);
        if (writeback && rn == r) {
            return true;
        }
        if (!vec && (((i >> 27) & 7) == 1 ||                             // exclusives
                     (((i >> 27) & 7) == 7 && ((i >> 21) & 1) && ((i >> 10) & 3) == 0))) { // atomics
            return true;
        }
        if (vec) {
            return false;
        }
        const bool load = pair ? ((i >> 22) & 1) : (((i >> 22) & 3) != 0 || ((i >> 27) & 7) == 3);
        if (!load) {
            return false;
        }
        return rd == r || (pair && rt2 == r);
    }
    if ((op0 & 0x7) == 0x7) { // SIMD and FP data processing: writes vector registers, except
        if ((i & 0x5F000000) == 0x1E000000 && (((i >> 21) & 1) == 0 || ((i >> 10) & 0x3F) == 0)) {
            return rd == r; // conversions between FP and integer or fixed point (FMOV, FCVTZS, ...)
        }
        if ((i & 0x9FE08400) == 0x0E000400) {
            return rd == r; // SIMD copy (SMOV, UMOV to a general register)
        }
        return false;
    }
    return rd == r; // integer data processing
}

// Nearest instruction before `from` (at or after `begin`, within `window`
// instructions) that may write `r`. Walks straight back, except into a block
// with no fallthrough predecessor (the instruction before it is B/BR/RET):
// there it continues from the nearest direct branch to the block that precedes
// it (a forward edge, not a loop's back edge), within [begin, end) and 64 KB.
// The nearest direct branch (not BL) before `p`, within [begin, p) and 64 KB, that targets `p`; 0 if none.
u64 BranchSourceTo(const Image& img, u64 begin, u64 p) {
    const u64 lo = std::max(begin, p > 0x10000 ? p - 0x10000 : 0);
    for (u64 q = p - 4; q >= lo && q + 4 > lo; q -= 4) {
        u64 t;
        const u32 i = img.Word(q);
        if ((i & 0xFC000000) != 0x94000000 && DirectTarget(i, q, t) && t == p) {
            return q;
        }
    }
    return 0;
}

bool FindDef(const Image& img, u64 begin, u64 from, u32 r, u64& at, u32 window = 64,
             bool follow = false) {
    for (u64 p = from; p > begin && window--;) {
        if (follow && IsUnconditionalTransfer(img.Word(p - 4))) {
            const u64 src = BranchSourceTo(img, begin, p);
            if (!src) {
                return false;
            }
            p = src; // the branch itself writes no register
            continue;
        }
        p -= 4;
        if (MayWrite(img.Word(p), r)) {
            at = p;
            return true;
        }
    }
    return false;
}

// Address in `r` at `from`: ADR, or ADD #lo12 of an ADRP page. First by the
// definition FindDef reaches; failing that, by the nearest preceding ADR or
// ADD-of-ADRP writing `r` anywhere before `from` (registers hoisted out of
// loops). Register copies (MOV) are followed. Loose on purpose: the emitted
// switch is keyed on the computed target, so a wrong table can only miss
// cases, never misdirect control.
bool AddressIn(const Image& img, u64 begin, u64 from, u32 r, u64& addr,
               u64* adr_pc = nullptr, int depth = 0) {
    auto decode = [&](u64 p) {
        const u32 i = img.Word(p);
        if ((i & 0xFFE0FFE0) == 0xAA0003E0 && (i & 31) == r && depth < 4) { // mov xr, xs: a copy
            return AddressIn(img, begin, p, (i >> 16) & 31, addr, adr_pc, depth + 1);
        }
        if (IsAdr(i) && (i & 31) == r) {
            addr = p + static_cast<u64>(AdrImm(i));
            if (adr_pc) {
                *adr_pc = p;
            }
            return true;
        }
        if (IsAddImm64(i) && (i & 31) == r) {
            u64 q;
            if (!FindDef(img, begin, p, (i >> 5) & 31, q, 2048, true) || !IsAdrp(img.Word(q))) {
                return false;
            }
            addr = (q & ~0xFFFULL) + static_cast<u64>(AdrImm(img.Word(q)) << 12) + ((i >> 10) & 0xFFF);
            return true;
        }
        return false;
    };
    u64 p;
    if (FindDef(img, begin, from, r, p, 2048, true) && decode(p)) {
        return true;
    }
    for (u64 q = from, n = 0; q > begin && n < 4096; ++n) {
        q -= 4;
        const u32 i = img.Word(q);
        if ((IsAdr(i) || IsAddImm64(i)) && (i & 31) == r) {
            return decode(q);
        }
    }
    return false;
}

// TableBound result flag: the count is loose (from a mask on the index, or a
// compare of another register), so the table may be shorter than it allows.
constexpr u64 kMaskBound = 1ULL << 63;

// The value of register `r` before `at`, traced back through register moves
// and zero-extending masks (mov, and #2^n-1, uxtb/uxth) to the instruction
// that really defines it. Two registers hold the same index if their roots
// match. `mask` receives the tightest mask seen (~0 if none).
struct ValueRoot {
    u32 reg;
    u64 def; // 0: not defined in the window (a parameter, or out of reach)
    bool operator==(const ValueRoot&) const = default;
};
ValueRoot Root(const Image& img, u64 begin, u64 at, u32 r, u64* mask = nullptr) {
    if (mask) {
        *mask = ~0ULL;
    }
    for (int depth = 0; depth < 8; ++depth) {
        u64 p;
        if (!FindDef(img, begin, at, r, p, 256, true)) {
            return {r, 0};
        }
        const u32 i = img.Word(p);
        const u32 rn = (i >> 5) & 31, immr = (i >> 16) & 63, imms = (i >> 10) & 63;
        const bool sf = i >> 31, n = (i >> 22) & 1;
        u64 m = ~0ULL;
        if ((i & 0x7FE0FFE0) == 0x2A0003E0) { // mov (orr rd, zr, rm)
            m = sf ? ~0ULL : 0xFFFFFFFFULL;
            r = (i >> 16) & 31;
        } else if (((i & 0x7F800000) == 0x12000000 || (i & 0x7F800000) == 0x53000000) && immr == 0 &&
                   ((sf && n) || (!sf && !n && imms < 31))) { // and #2^n-1 / ubfm #0 (uxtb, uxth)
            m = (2ULL << imms) - 1;
            r = rn;
        } else if ((i & 0x7F800000) == 0x53000000 && imms >= immr) { // ubfx / lsr: a bit field
            if (mask) {
                *mask &= (2ULL << (imms - immr)) - 1;
            }
            return {r, p};
        } else {
            return {r, p};
        }
        if (mask) {
            *mask &= m;
        }
        at = p;
    }
    return {r, 0};
}

// Number of cases from the bound check on the index register `k` at `load`:
//   cmp/subs x, #N ; b.hi default | b.hs ...   in the instructions before the load, or
//   cmp/subs x, #N ; b.ls table                anywhere in [begin, end), jumping just before the load,
// with x holding the same value as k (see Root). "Before" follows a branch into
// the load's block, as compilers duplicate the b.hi + table tail into each
// predecessor (cmp ; tbz ... ; b.hi default ; table / target: b.hi default ; table).
// Failing that, an index masked to 2^n-1 has 2^n cases; failing that, a CMP of
// another register guarding the block gives a loose count (kMaskBound), and
// failing everything, the table is read until an entry cannot be a case.
u64 TableBound(const Image& img, u64 begin, u64 end, u64 load, u32 k) {
    u64 mask;
    const ValueRoot index = Root(img, begin, load, k, &mask);
    auto cmp_n = [&](u64 p, u64& n) {
        const u32 i = img.Word(p);
        if ((i & 0x7FC00000) == 0x71000000 && Root(img, begin, p, (i >> 5) & 31) == index) {
            n = (i >> 10) & 0xFFF; // SUBS (cmp) _, x, #N
            return true;
        }
        // CCMP x, #N, #nzcv, cond with nzcv = C (2): when skipped, "hi" holds.
        if ((i & 0x7FE00C10) == 0x7A400800 && (i & 15) == 2 &&
            Root(img, begin, p, (i >> 5) & 31) == index) {
            n = (i >> 16) & 31;
            return true;
        }
        return false;
    };
    auto count_for = [](u32 cond, u64 n) -> u64 {
        switch (cond) {
        case 8: case 9: return n + 1; // hi / ls
        case 2: case 3: return n;     // hs / lo
        default: return 0;
        }
    };
    int cond = -1; // the last B.cond before the load on the path walked back
    u64 loose = 0;
    for (u64 p = load, steps = 0; p > begin && steps < 32; ++steps) {
        if (IsUnconditionalTransfer(img.Word(p - 4))) {
            p = BranchSourceTo(img, begin, p); // the branch itself is not the bound's B.cond
            if (!p) {
                break;
            }
            continue;
        }
        p -= 4;
        const u32 i = img.Word(p);
        if ((i & 0xFF000010) == 0x54000000) {
            if (cond < 0) {
                cond = static_cast<int>(i & 15);
            }
            continue;
        }
        u64 n;
        if (cmp_n(p, n)) {
            if (cond >= 0) {
                if (const u64 c = count_for(static_cast<u32>(cond), n)) {
                    return c;
                }
            }
            break;
        }
        if ((i & 0x7FC0001F) == 0x7100001F && cond >= 0) {
            // CMP of another register guarding this block: the index is usually derived
            // from it (extended, hoisted), so its bound is likely the case count. Loose,
            // like a mask: the table is cut at the first entry that cannot be a case.
            loose = count_for(static_cast<u32>(cond), (i >> 10) & 0xFFF);
            break;
        }
    }
    // A b.ls into the table block from elsewhere in the function.
    for (u64 p = begin; p + 8 <= end; p += 4) {
        const u32 b = img.Word(p + 4);
        u64 t = 0, n;
        if ((b & 0xFF00001F) == 0x54000009 && DirectTarget(b, p + 4, t) && t <= load &&
            load - t <= 24 && cmp_n(p, n)) {
            return n + 1;
        }
    }
    if (mask < 4096) {
        return (mask + 1) | kMaskBound;
    }
    if (loose) {
        return loose | kMaskBound;
    }
    // No bound found (an index bounded by its caller, or through arithmetic):
    // read entries until one cannot be a case. Extra cases are harmless.
    return 256 | kMaskBound;
}

// Recovers the case targets of a jump table ending at `br`. Idioms (any
// instructions may be interleaved):
//   GCC:   ldrb wi, [xt, wk, uxtw] ; adr xb, base ; add xd, xb, wi, sxtb #2 ; br xd
//   clang: adr xb, base ; ldr{b,h} wi, [xt, xk{, lsl #s}] ; add xd, xb, xi, lsl #2 ; br xd
//   clang: ldrsw xi, [xt, xk, lsl #2] ; add xd, xi, xt ; br xd      (base = table)
// with xt from ADRP+ADD and a bound check on the index (TableBound). Returns
// false when anything is not found, or a target would leave [begin, end).
// `base_adr` receives the PC of the ADR that loads the base, if any.
bool RecoverJumpTable(const Image& img, u64 begin, u64 end, u64 br, std::vector<u64>& targets,
                      u64* base_adr = nullptr) {
    const char* dbg_env = std::getenv("NXR_DEBUG_JT");
    const bool dbg = dbg_env && (!std::strcmp(dbg_env, "all") || std::strtoull(dbg_env, nullptr, 16) == br);
#define JT_FAIL do { if (dbg) std::fprintf(stderr, "jt fail line %d at %llx\n", __LINE__, (unsigned long long)br); return false; } while (0)
    const u32 xd = (img.Word(br) >> 5) & 31;
    u64 add_pc;
    if (!FindDef(img, begin, br, xd, add_pc)) {
        JT_FAIL;
    }
    const u32 add = img.Word(add_pc);
    u32 shift = 0;
    int ext_bits = 0; // the add's own sign extension of the entry (sxtb/sxth/sxtw)
    if ((add & 0xFFE00000) == 0x8B200000) { // ADD (extended register)
        const u32 option = (add >> 13) & 7;
        shift = (add >> 10) & 7;
        ext_bits = option >= 4 && option < 7 ? 8 << (option - 4) : 0;
    } else if ((add & 0xFFE00000) == 0x8B000000) { // ADD (shifted register), LSL only
        shift = (add >> 10) & 63;
    } else {
        JT_FAIL;
    }
    const u32 rn = (add >> 5) & 31, rm = (add >> 16) & 31;

    // Which operand is the loaded entry?
    auto is_table_load = [&](u32 r, u64& at) {
        if (!FindDef(img, begin, add_pc, r, at)) {
            return false;
        }
        const u32 i = img.Word(at);
        return (i & 0x3B200C00) == 0x38200800 && (i & 31) == r && ((i >> 22) & 3) != 0 &&
               (i >> 30) <= 2; // LDR{B,H,,SB,SH,SW} (register offset)
    };
    u64 load = 0;
    u32 base_reg;
    if (is_table_load(rm, load)) {
        base_reg = rn;
    } else if (shift == 0 && is_table_load(rn, load)) {
        base_reg = rm;
    } else {
        JT_FAIL;
    }
    const u32 ld = img.Word(load);
    const u32 width = 1u << (ld >> 30), xt = (ld >> 5) & 31, xk = (ld >> 16) & 31;
    const bool sign = ((ld >> 22) & 3) >= 2;
    if (((ld >> 12) & 1) == 0 && width != 1) {
        JT_FAIL; // unscaled index into a multi-byte table
    }
    u64 table, base;
    if (!AddressIn(img, begin, load, xt, table) ||
        !AddressIn(img, begin, add_pc, base_reg, base, base_adr)) {
        JT_FAIL;
    }
    const u64 bound = TableBound(img, begin, end, load, xk);
    const bool masked = bound & kMaskBound;
    u64 count = bound & ~kMaskBound;
    if (masked && table + count * width > img.mem.size()) {
        count = (img.mem.size() - std::min<u64>(table, img.mem.size())) / width;
    }
    if (count == 0 || count > 4096 || table + count * width > img.mem.size()) {
        JT_FAIL;
    }
    targets.clear();
    for (u64 n = 0; n < count; ++n) {
        u64 e = 0;
        std::memcpy(&e, &img.mem[table + n * width], width);
        s64 v = static_cast<s64>(e);
        if (sign) {
            v = static_cast<s64>(e << (64 - 8 * width)) >> (64 - 8 * width);
        }
        if (ext_bits) {
            v = static_cast<s64>(static_cast<u64>(v) << (64 - ext_bits)) >> (64 - ext_bits);
        }
        const u64 t = base + static_cast<u64>(v << shift);
        if (t < begin || t >= end || (t & 3)) {
            // A mask only bounds the index: keep the cases before the first
            // entry that cannot be one (an extra in-function label is harmless).
            if (masked && targets.size() >= 2) {
                break;
            }
            JT_FAIL;
        }
        targets.push_back(t);
    }
    return true;
#undef JT_FAIL
}

// The ADRs that load jump-table bases: an ADR feeding the ADD before a BR.
// Found before function ranges exist, and whether or not the table is
// recovered. Their targets are case labels, not function starts.
std::set<u64> JumpTableBaseAdrs(const Image& img) {
    std::set<u64> out;
    for (u64 pc = img.text_begin + 4; pc + 4 <= img.text_end; pc += 4) {
        const u32 i = img.Word(pc);
        u64 add_pc;
        if (!IsBr(i) || ((i >> 5) & 31) == 30 ||
            !FindDef(img, img.text_begin, pc, (i >> 5) & 31, add_pc)) {
            continue;
        }
        const u32 add = img.Word(add_pc);
        if ((add & 0xFF200000) != 0x8B200000 && (add & 0xFF200000) != 0x8B000000) {
            continue;
        }
        for (u32 r : {(add >> 5) & 31, (add >> 16) & 31}) {
            u64 at;
            if (FindDef(img, img.text_begin, add_pc, r, at) && IsAdr(img.Word(at))) {
                out.insert(at);
            }
        }
    }
    return out;
}

void CollectAddressSeeds(Analysis& a, const Image& img) {
    // ADRP Xd, page followed by ADD Xd, Xd, #lo12 (anywhere later, until Xd is
    // overwritten), and ADR Xd, label. Deliberately loose: a data address that
    // happens to land in .text only adds a spurious split, which the emitter
    // turns into a tail call.
    const std::set<u64> table_bases = JumpTableBaseAdrs(img);
    std::array<u64, 32> page;
    page.fill(~0ULL);
    for (u64 pc = img.text_begin; pc + 4 <= img.text_end; pc += 4) {
        const u32 i = img.Word(pc);
        const u32 rd = i & 31;
        if (IsAdrp(i)) {
            page[rd] = (pc & ~0xFFFULL) + static_cast<u64>(AdrImm(i) << 12);
        } else if (i == 0x94000002 && pc + 12 <= img.text_end &&
                   (img.Word(pc + 8) & 0xFFFFFFE0) == 0xB94003C0) {
            // rtld's pre-relocation addressing: bl 1f; .word off; 1: ldr wN, [x30];
            // sxtw; add xN, xN, x30 -> address = literal + off (its callbacks).
            AddSeed(a, img, pc + 4 + static_cast<u64>(static_cast<s64>(static_cast<s32>(img.Word(pc + 4)))),
                    kSeedAddress);
        } else if (IsAdr(i)) {
            if (!table_bases.count(pc)) {
                AddSeed(a, img, pc + static_cast<u64>(AdrImm(i)), kSeedAddress);
            }
            page[rd] = ~0ULL;
        } else if (IsAddImm64(i)) {
            const u32 rn = (i >> 5) & 31;
            if (page[rn] != ~0ULL) {
                AddSeed(a, img, page[rn] + ((i >> 10) & 0xFFF), kSeedAddress);
            }
            if (rd != rn) {
                page[rd] = ~0ULL;
            }
        } else {
            page[rd] = ~0ULL;
        }
    }
}

// Counted as an unrecovered table: a BR whose register comes from an ADD of a
// table load (the shape RecoverJumpTable starts from).
bool LooksLikeJumpTable(const Image& img, u64 begin, u64 br) {
    u64 add_pc;
    const u32 xd = (img.Word(br) >> 5) & 31;
    if (!FindDef(img, begin, br, xd, add_pc)) {
        return false;
    }
    const u32 add = img.Word(add_pc);
    if ((add & 0xFF200000) != 0x8B200000 && (add & 0xFF200000) != 0x8B000000) {
        return false;
    }
    for (u32 r : {(add >> 5) & 31, (add >> 16) & 31}) {
        u64 at;
        if (FindDef(img, begin, add_pc, r, at) && (img.Word(at) & 0x3B200C00) == 0x38200800) {
            return true;
        }
    }
    return false;
}

} // namespace

bool IsUnconditionalTransfer(u32 i) {
    return (i & 0xFC000000) == 0x14000000 ||   // B
           (i & 0xFFFFFC1F) == 0xD61F0000 ||   // BR
           (i & 0xFFFFFC1F) == 0xD65F0000;     // RET
}

bool DirectTarget(u32 i, u64 pc, u64& out) {
    if ((i & 0x7C000000) == 0x14000000) { // B, BL
        out = pc + static_cast<u64>(static_cast<s64>(static_cast<s32>(i << 6) >> 6) * 4);
        return true;
    }
    if ((i & 0xFF000010) == 0x54000000 || (i & 0x7E000000) == 0x34000000) { // B.cond, CBZ/CBNZ
        out = pc + static_cast<u64>(static_cast<s64>(static_cast<s32>(((i >> 5) & 0x7FFFF) << 13) >> 13) * 4);
        return true;
    }
    if ((i & 0x7E000000) == 0x36000000) { // TBZ/TBNZ
        out = pc + static_cast<u64>(static_cast<s64>(static_cast<s32>(((i >> 5) & 0x3FFF) << 18) >> 18) * 4);
        return true;
    }
    return false;
}

const Function* Analysis::Find(u64 off) const {
    auto it = std::upper_bound(functions.begin(), functions.end(), off,
                               [](u64 v, const Function& f) { return v < f.begin; });
    if (it == functions.begin()) {
        return nullptr;
    }
    --it;
    return off < it->end ? &*it : nullptr;
}

Analysis Discover(const Image& img, const DiscoveryOptions& opts) {
    Analysis a;
    AddSeed(a, img, img.entry, kSeedEntry);
    for (const auto& s : img.dynsyms) {
        if (s.defined && s.function && opts.use_export_seeds) {
            AddSeed(a, img, s.value, kSeedExport);
        }
    }
    for (const auto& r : img.relocations) {
        u64 target = 0;
        if (r.type == R_AARCH64_RELATIVE) {
            target = static_cast<u64>(r.addend);
        } else if ((r.type == R_AARCH64_ABS64 || r.type == R_AARCH64_GLOB_DAT ||
                    r.type == R_AARCH64_JUMP_SLOT) &&
                   r.symbol < img.dynsyms.size() && img.dynsyms[r.symbol].defined) {
            target = img.dynsyms[r.symbol].value + static_cast<u64>(r.addend);
        } else {
            continue;
        }
        AddSeed(a, img, target, kSeedReloc);
    }
    // A JUMP_SLOT initially holds the module-relative address of PLT0, the
    // lazy-binding entry: reached only through the GOT, so seed it.
    for (const auto& r : img.relocations) {
        if (r.type == R_AARCH64_JUMP_SLOT && r.offset + 8 <= img.mem.size()) {
            AddSeed(a, img, img.Dword(r.offset), kSeedReloc);
        }
    }
    for (u64 f : img.fde_starts) {
        AddSeed(a, img, f, kSeedFde);
    }
    for (u64 pc = img.text_begin; pc + 4 <= img.text_end; pc += 4) {
        u64 t;
        if (IsBl(img.Word(pc)) && DirectTarget(img.Word(pc), pc, t)) {
            AddSeed(a, img, t, kSeedCall);
        }
    }
    if (opts.use_address_seeds) {
        CollectAddressSeeds(a, img);
    }
    for (u64 s : opts.extra_seeds) {
        AddSeed(a, img, s, kSeedExtra);
    }
    for (u64 s : opts.linked_seeds) {
        AddSeed(a, img, s, kSeedLinked);
    }

    // Ranges run from one seed to the next. A direct branch that leaves its
    // range must be a tail call, so its target becomes a seed too; repeat
    // until no range changes.
    for (bool changed = true; changed;) {
        changed = false;
        std::vector<u64> starts;
        for (const auto& [off, _] : a.seeds) {
            starts.push_back(off);
        }
        a.jump_tables.clear();
        a.unresolved_jump_tables = 0;
        a.unresolved_jump_table_sites.clear();
        for (size_t n = 0; n < starts.size(); ++n) {
            const u64 begin = starts[n];
            const u64 end = n + 1 < starts.size() ? starts[n + 1] : img.text_end;
            for (u64 pc = begin; pc < end; pc += 4) {
                const u32 i = img.Word(pc);
                if (IsBr(i) && ((i >> 5) & 31) != 30) {
                    std::vector<u64> targets;
                    if (RecoverJumpTable(img, begin, end, pc, targets)) {
                        a.jump_tables[pc] = std::move(targets);
                    } else if (LooksLikeJumpTable(img, begin, pc)) {
                        ++a.unresolved_jump_tables; // looks like a table, but not recovered
                        a.unresolved_jump_table_sites.push_back(pc);
                    }
                }
                u64 t;
                if (!IsBl(i) && DirectTarget(i, pc, t) && (t < begin || t >= end) &&
                    img.InText(t) && !a.seeds.count(t)) {
                    a.seeds[t] |= kSeedTailCall;
                    changed = true;
                }
            }
        }
    }

    std::vector<u64> starts;
    for (const auto& [off, _] : a.seeds) {
        starts.push_back(off);
    }
    for (size_t n = 0; n < starts.size(); ++n) {
        u64 end = n + 1 < starts.size() ? starts[n + 1] : img.text_end;
        // Trailing padding belongs to no function.
        while (end > starts[n] + 4 && IsPadding(img.Word(end - 4))) {
            end -= 4;
        }
        a.functions.push_back({starts[n], end, a.seeds[starts[n]]});
    }
    return a;
}

std::string ValidateAgainstSymbols(const Image& img, const Analysis& a) {
    std::ostringstream o;
    std::set<u64> truth;
    for (const auto& [addr, name] : img.symbols) {
        if (img.InText(addr)) {
            truth.insert(addr);
        }
    }
    if (truth.empty()) {
        return "no .symtab function symbols to validate against\n";
    }
    std::set<u64> found;
    for (const auto& f : a.functions) {
        found.insert(f.begin);
    }
    size_t hit = 0;
    for (u64 t : truth) {
        hit += found.count(t);
    }
    o << "symbols in .text: " << truth.size() << ", discovered functions: " << found.size() << "\n";
    o << "recall (symbols found as function starts): " << hit << "/" << truth.size() << "\n";
    o << "precision (starts that are symbols): " << hit << "/" << found.size() << "\n";

    const std::pair<SeedSource, const char*> names[] = {
        {kSeedEntry, "entry"}, {kSeedExport, "export"}, {kSeedReloc, "reloc"},
        {kSeedCall, "call"}, {kSeedFde, "fde"}, {kSeedAddress, "address"},
        {kSeedTailCall, "tailcall"}, {kSeedExtra, "extra"}, {kSeedLinked, "linked"}};
    o << "per source (seeds / of which symbols):\n";
    for (auto [bit, name] : names) {
        size_t n = 0, good = 0;
        for (const auto& [off, src] : a.seeds) {
            if (src & bit) {
                ++n;
                good += truth.count(off);
            }
        }
        o << "  " << name << ": " << n << " / " << good << "\n";
    }
    // Starts strictly inside an FDE's range split a known function (or the FDE
    // covers several functions, which compilers don't normally emit).
    if (!img.fde_ranges.empty()) {
        size_t inside = 0, shown_split = 0;
        std::ostringstream ex;
        for (const auto& [b, e] : img.fde_ranges) {
            for (auto it = found.upper_bound(b); it != found.end() && *it < e; ++it) {
                ++inside;
                if (shown_split++ < 5) {
                    ex << " 0x" << std::hex << *it << " (FDE 0x" << b << "-0x" << e << ")" << std::dec;
                }
            }
        }
        o << "starts inside an FDE range (likely false splits): " << inside << " of "
          << img.fde_ranges.size() << " FDEs parsed;" << ex.str() << "\n";
    }
    o << "unresolved jump tables: " << a.unresolved_jump_tables;
    for (size_t n = 0; n < a.unresolved_jump_table_sites.size() && n < 8; ++n) {
        o << (n ? ", 0x" : " (e.g. 0x") << std::hex << a.unresolved_jump_table_sites[n] << std::dec;
    }
    o << (a.unresolved_jump_table_sites.empty() ? "\n" : ")\n");
    size_t shown = 0;
    for (u64 t : truth) {
        if (!found.count(t) && shown++ < 15) {
            const Function* f = a.Find(t);
            o << "  missed " << img.symbols.at(t) << " @0x" << std::hex << t << " (inside 0x"
              << (f ? f->begin : 0) << ")" << std::dec << "\n";
        }
    }
    shown = 0;
    for (u64 s : found) {
        if (!truth.count(s) && shown++ < 15) {
            auto it = img.symbols.upper_bound(s);
            const std::string owner = it == img.symbols.begin() ? "?" : std::prev(it)->second;
            o << "  extra start @0x" << std::hex << s << std::dec << " (inside " << owner
              << ", sources 0x" << std::hex << a.seeds.at(s) << std::dec << ")\n";
        }
    }
    return o.str();
}

} // namespace nxrecomp
