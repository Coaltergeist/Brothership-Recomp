// Integer Advanced SIMD as host vector code. Phase 1 covers the vector (not scalar)
// forms of: three same (halving / rounding halving add and sub, add, sub, mul, mla, mls, compares,
// max, min, abd, aba, sqdmulh, sqrdmulh), shift by immediate (sshr, ushr, ssra, usra, srshr, urshr,
// srsra, ursra, shl, sshll, ushll, shrn, rshrn and the saturating narrowing shifts), three different
// (addl, addw, subl, subw, mlal, mlsl, mull, abdl, abal) and two-register misc (xtn, sqxtn, uqxtn,
// sqxtun, abs, neg, addlp, adalp, compares with zero). Phase 2 / Phase 4 step 2 add the moves, permutes
// and logic (and, bic, orr, orn, eor, bsl, bit, bif, not, dup, ins, umov, smov, ext, zip, uzp, trn, rev,
// movi, mvni, orr / bic immediate, fmov vector immediate) and the scalar FP moves that are exact by nature
// (fmov register / immediate / general, fabs, fneg, fcsel; vector fabs, fneg). Every other encoding,
// reserved ones included, is declined and keeps drippu's translation.
//
// Each instruction loads its operands from c->vreg into vector variables, computes, and stores: a
// 64-bit form (Q = 0) is computed on the whole register and stores the low half and zeroes the high
// half; saturation masks then cover only the low half's lanes. Arithmetic that may wrap is done on
// unsigned lanes (signed overflow is undefined in C); halving and rounding never widen past the lane
// (no overflow); shifts by the whole lane width, legal in A64, are special-cased (undefined in C).
// tools/difftest --full --simd checks every encoding this accepts against a Cortex-A57.
#include "simd.h"

#include <string>
#include <vector>

namespace nxrecomp::simd {
namespace {

using std::string;
using std::to_string;

/// A vector type: lane signedness, lane bits, total bytes (8, 16 or 32).
string T(bool s, unsigned bits, unsigned bytes) {
    return string("nxv_") + (s ? "s" : "u") + to_string(bits) + "x" + to_string(bytes * 8 / bits);
}

string Reg(unsigned r) {
    return "c->vreg[" + to_string(r) + "]";
}

/// "T v; memcpy(&v, <register> + off, sizeof v);"
string Load(const string& type, const string& v, unsigned r, unsigned off = 0) {
    return type + " " + v + "; memcpy(&" + v + ", (const uint8_t*)" + Reg(r) + (off ? " + " + to_string(off) : "") +
           ", sizeof " + v + "); ";
}

/// Stores a 16-byte result: whole register (Q = 1), or the low half and a zeroed high half (Q = 0).
string Store(unsigned rd, const string& v, unsigned q) {
    if (q) return "memcpy(" + Reg(rd) + ", &" + v + ", 16); ";
    return "memcpy(" + Reg(rd) + ", &" + v + ", 8); " + Reg(rd) + "[1] = 0; ";
}

/// Stores an 8-byte narrowed result: low half and zeroed high half (Q = 0), or the high half alone,
/// keeping the low half (the "2" forms, Q = 1).
string StoreNarrow(unsigned rd, const string& v, unsigned q) {
    if (q) return "memcpy((uint8_t*)" + Reg(rd) + " + 8, &" + v + ", 8); ";
    return "memcpy(" + Reg(rd) + ", &" + v + ", 8); " + Reg(rd) + "[1] = 0; ";
}

string Qc(const string& mask, unsigned bytes) {
    return "c->fpsr |= nxv_qc(&" + mask + ", " + to_string(bytes) + "); ";
}

/// A scalar constant of a lane type (64-bit lanes need the suffix).
string K(long long v, bool s, unsigned bits) {
    if (bits == 64) return s ? to_string(v) + "LL" : to_string(static_cast<unsigned long long>(v)) + "ULL";
    return to_string(v);
}

long long Max(bool s, unsigned bits) {
    if (bits == 64) return s ? 0x7fffffffffffffffLL : -1LL;
    return s ? (1LL << (bits - 1)) - 1 : (1LL << bits) - 1;
}
long long Min(bool s, unsigned bits) {
    if (!s) return 0;
    return bits == 64 ? (-0x7fffffffffffffffLL - 1) : -(1LL << (bits - 1));
}

/// x >> sh with rounding (adding bit sh - 1), for 1 <= sh <= lane bits, in x's own type: no overflow.
string RoundShr(const string& x, const string& type, bool s, unsigned bits, unsigned sh) {
    if (sh == bits) return s ? "(" + type + "){0}" : "((" + x + " >> " + to_string(bits - 1) + ") & 1)";
    return "((" + x + " >> " + to_string(sh) + ") + ((" + x + " >> " + to_string(sh - 1) + ") & 1))";
}

/// x >> sh, 1 <= sh <= lane bits.
string Shr(const string& x, const string& type, bool s, unsigned bits, unsigned sh) {
    if (sh == bits) return s ? "(" + x + " >> " + to_string(bits - 1) + ")" : "(" + type + "){0}";
    return "(" + x + " >> " + to_string(sh) + ")";
}

/// Clamps `x` (type w, signed if ws) to [lo, hi] into `r`, the clamped lanes' mask into `m`.
string Clamp(const string& x, const string& w, bool ws, unsigned wbits, long long lo, long long hi, bool lo_needed,
             bool lo_s) {
    string s = w + " m_hi = (" + w + ")(" + x + " > " + K(hi, ws, wbits) + "); ";
    if (lo_needed) {
        s += w + " m_lo = (" + w + ")(" + x + " < " + K(lo, lo_s, wbits) + "); ";
        s += w + " m = m_hi | m_lo; ";
        s += w + " r = (" + x + " & ~m) | (m_hi & " + K(hi, ws, wbits) + ") | (m_lo & " + K(lo, ws, wbits) + "); ";
    } else {
        s += w + " m = m_hi; ";
        s += w + " r = (" + x + " & ~m) | (m_hi & " + K(hi, ws, wbits) + "); ";
    }
    return s;
}

// --- three same: 0 Q U 01110 size 1 Rm opcode 1 Rn Rd -----------------------------------------------
string ThreeSame(unsigned q, unsigned u, unsigned size, unsigned op, unsigned rd, unsigned rn, unsigned rm) {
    const unsigned e = 8u << size;
    if (size == 3 && !q) return ""; // .1d: reserved in the vector forms
    const string S = T(true, e, 16), U = T(false, e, 16), A = u ? U : S;
    string b = Load(A, "a", rn) + Load(A, "b", rm);
    auto store = [&](const string& expr) { return b + U + " r = (" + U + ")(" + expr + "); " + Store(rd, "r", q); };
    switch (op) {
    case 0x10: return store(u ? "(" + U + ")a - (" + U + ")b" : "(" + U + ")a + (" + U + ")b"); // ADD / SUB
    case 0x13: // MUL (PMUL: U = 1)
        if (u || size == 3) return "";
        return store("(" + U + ")a * (" + U + ")b");
    case 0x12: // MLA / MLS
        if (size == 3) return "";
        b += Load(U, "d", rd);
        return store(string("d ") + (u ? "-" : "+") + " (" + U + ")a * (" + U + ")b");
    case 0x00: // SHADD / UHADD
        if (size == 3) return "";
        return store("(" + U + ")(a >> 1) + (" + U + ")(b >> 1) + (" + U + ")(a & b & 1)");
    case 0x02: // SRHADD / URHADD
        if (size == 3) return "";
        return store("(" + U + ")(a >> 1) + (" + U + ")(b >> 1) + (" + U + ")((a | b) & 1)");
    case 0x04: // SHSUB / UHSUB: floor((a - b) / 2)
        if (size == 3) return "";
        return store("(" + U + ")(a >> 1) - (" + U + ")(b >> 1) - (" + U + ")(~a & b & 1)");
    case 0x06: return store("a > b");   // CMGT / CMHI
    case 0x07: return store("a >= b");  // CMGE / CMHS
    case 0x11: return store(u ? "a == b" : "(a & b) != 0"); // CMEQ / CMTST
    case 0x0c: // SMAX / UMAX
    case 0x0d: // SMIN / UMIN
        if (size == 3) return "";
        b += A + " m = (" + A + ")(a " + (op == 0x0c ? ">" : "<") + " b); ";
        return store("(a & m) | (b & ~m)");
    case 0x0e: // SABD / UABD: max - min, as unsigned
    case 0x0f: // SABA / UABA
        if (size == 3) return "";
        b += A + " m = (" + A + ")(a > b); " + U + " ad = (" + U + ")((a & m) | (b & ~m)) - (" + U +
             ")((b & m) | (a & ~m)); ";
        if (op == 0x0e) return store("ad");
        b += Load(U, "d", rd);
        return store("d + ad");
    case 0x16: { // SQDMULH / SQRDMULH: sat((2ab [+ 2^(e-1)]) >> e); only min * min saturates
        if (size != 1 && size != 2) return "";
        const string W = T(true, 2 * e, 32);
        b = Load(S, "a", rn) + Load(S, "b", rm);
        b += W + " p = __builtin_convertvector(a, " + W + ") * __builtin_convertvector(b, " + W + "); ";
        b += W + " t = " + (u ? "(p + " + K(1LL << (e - 2), true, 2 * e) + ")" : string("p")) + " >> " +
             to_string(e - 1) + "; ";
        b += W + " mw = (" + W + ")(t > " + K(Max(true, e), true, 2 * e) + "); ";
        b += S + " m = __builtin_convertvector(mw, " + S + "); ";
        b += S + " r = (__builtin_convertvector(t, " + S + ") & ~m) | (m & " + K(Max(true, e), true, e) + "); ";
        return b + Qc("m", q ? 16 : 8) + Store(rd, "r", q);
    }
    default: return "";
    }
}

// --- three different: 0 Q U 01110 size 1 Rm opcode 00 Rn Rd (Q: the "2" forms read the high halves) ---
string ThreeDiff(unsigned q, unsigned u, unsigned size, unsigned op, unsigned rd, unsigned rn, unsigned rm) {
    if (size == 3) return "";
    const unsigned e = 8u << size, off = q ? 8 : 0;
    const bool s = !u;
    const string N = T(s, e, 8), W = T(s, 2 * e, 16), WU = T(false, 2 * e, 16);
    const string wa = "__builtin_convertvector(an, " + W + ")", wb = "__builtin_convertvector(bn, " + W + ")";
    string b;
    auto store = [&](const string& expr) { return b + WU + " r = (" + WU + ")(" + expr + "); " + Store(rd, "r", 1); };
    switch (op) {
    case 0x0: case 0x2: // SADDL / UADDL, SSUBL / USUBL
        b = Load(N, "an", rn, off) + Load(N, "bn", rm, off);
        return store("(" + WU + ")" + wa + (op == 0x0 ? " + " : " - ") + "(" + WU + ")" + wb);
    case 0x1: case 0x3: // SADDW / UADDW, SSUBW / USUBW
        b = Load(W, "aw", rn) + Load(N, "bn", rm, off);
        return store("(" + WU + ")aw" + (op == 0x1 ? " + " : " - ") + "(" + WU + ")" + wb);
    case 0x8: case 0xa: case 0xc: // SMLAL / UMLAL, SMLSL / UMLSL, SMULL / UMULL (the product fits 2e bits)
        b = Load(N, "an", rn, off) + Load(N, "bn", rm, off);
        b += WU + " p = (" + WU + ")" + wa + " * (" + WU + ")" + wb + "; ";
        if (op == 0xc) return store("p");
        b += Load(WU, "d", rd);
        return store(op == 0x8 ? "d + p" : "d - p");
    case 0x7: case 0x5: // SABDL / UABDL, SABAL / UABAL
        b = Load(N, "an", rn, off) + Load(N, "bn", rm, off);
        b += W + " aw = " + wa + ", bw = " + wb + "; " + W + " m = (" + W + ")(aw > bw); ";
        b += WU + " ad = (" + WU + ")((aw & m) | (bw & ~m)) - (" + WU + ")((bw & m) | (aw & ~m)); ";
        if (op == 0x7) return store("ad");
        b += Load(WU, "d", rd);
        return store("d + ad");
    default: return "";
    }
}

string Rev(unsigned q, unsigned size, unsigned cbits, unsigned rd, unsigned rn); // below

// --- two-register misc: 0 Q U 01110 size 10000 opcode 10 Rn Rd ------------------------------------
string TwoMisc(unsigned q, unsigned u, unsigned size, unsigned op, unsigned rd, unsigned rn) {
    const unsigned e = 8u << size;
    switch (op) {
    case 0x12: case 0x14: { // XTN / SQXTUN, SQXTN / UQXTN: the destination lanes are e bits
        if (size == 3) return "";
        const bool src_s = !(op == 0x14 && u); // UQXTN reads unsigned lanes
        const string W = T(src_s, 2 * e, 16), WU = T(false, 2 * e, 16);
        string b = Load(W, "a", rn);
        if (op == 0x12 && !u) { // XTN: truncate
            b += T(false, e, 8) + " n = __builtin_convertvector((" + WU + ")a, " + T(false, e, 8) + "); ";
            return b + StoreNarrow(rd, "n", q);
        }
        const bool dst_s = op == 0x14 && !u; // SQXTN: signed result
        const long long lo = Min(dst_s, e), hi = Max(dst_s, e);
        b += Clamp("a", W, src_s, 2 * e, lo, hi, src_s, src_s);
        b += T(dst_s, e, 8) + " n = __builtin_convertvector(r, " + T(dst_s, e, 8) + "); ";
        return b + Qc("m", 16) + StoreNarrow(rd, "n", q);
    }
    case 0x0b: { // ABS / NEG (abs of the minimum stays the minimum)
        if (size == 3 && !q) return "";
        const string S = T(true, e, 16), U = T(false, e, 16);
        string b = Load(S, "a", rn);
        if (u) b += U + " r = (" + U + "){0} - (" + U + ")a; ";
        else b += U + " m = (" + U + ")(a < 0); " + U + " r = ((" + U + ")a ^ m) - m; ";
        return b + Store(rd, "r", q);
    }
    case 0x02: case 0x06: { // SADDLP / UADDLP, SADALP / UADALP: pairs of e-bit lanes into 2e-bit lanes
        if (size == 3) return "";
        const string WS = T(true, 2 * e, 16), WU = T(false, 2 * e, 16);
        string b = Load(WU, "w", rn);
        if (u) {
            b += WU + " sum = (w & " + K(Max(false, e), false, 2 * e) + ") + (w >> " + to_string(e) + "); ";
        } else {
            b += WU + " sum = (" + WU + ")((" + WS + ")(w << " + to_string(e) + ") >> " + to_string(e) + ") + (" +
                 WU + ")((" + WS + ")w >> " + to_string(e) + "); ";
        }
        if (op == 0x06) b += Load(WU, "d", rd) + "sum += d; ";
        return b + Store(rd, "sum", q);
    }
    case 0x0f: { // FABS / FNEG (vector): the sign bit, nothing else (size = 1 sz)
        if (size < 2 || (size == 3 && !q)) return "";
        const string U = T(false, e, 16), sign = size == 3 ? "0x8000000000000000ULL" : "0x80000000u";
        return Load(U, "a", rn) + U + " r = a " + (u ? "^ " : "& ~") + sign + "; " + Store(rd, "r", q);
    }
    case 0x00: // REV64 (U = 0), REV32 (U = 1)
        return Rev(q, size, u ? 32 : 64, rd, rn);
    case 0x01: // REV16
        return u ? "" : Rev(q, size, 16, rd, rn);
    case 0x05: { // NOT (U = 1, size 0)
        if (!u || size) return "";
        const string U = T(false, 64, 16);
        return Load(U, "a", rn) + U + " r = ~a; " + Store(rd, "r", q);
    }
    case 0x08: case 0x09: case 0x0a: { // CMGT / CMGE #0, CMEQ / CMLE #0, CMLT #0
        if (size == 3 && !q) return "";
        if (op == 0x0a && u) return "";
        static const char* const cmp[3][2] = {{">", ">="}, {"==", "<="}, {"<", ""}};
        const string S = T(true, e, 16);
        return Load(S, "a", rn) + S + " r = (" + S + ")(a " + cmp[op - 0x08][u] + " 0); " + Store(rd, "r", q);
    }
    default: return "";
    }
}

// --- Phase 2 / Phase 4 step 2: moves, permutes, logic and the exact FP moves (no FPSR effect) -----------

/// A general register as a source: register 31 reads as zero in these groups (c->x[31] holds SP).
string Gpr(unsigned r) {
    return r == 31 ? string("0ULL") : "c->x[" + to_string(r) + "]";
}

/// "Vn" as 16 bytes, element i of an e-bit-lane vector named v: "v[i]".
string Lane(const string& v, unsigned i) {
    return v + "[" + to_string(i) + "]";
}

/// A vector of `lanes` lanes from per-lane expressions (unused lanes 0).
string Init(const string& type, const std::vector<string>& v) {
    string s = "(" + type + "){";
    for (size_t k = 0; k < v.size(); ++k) s += (k ? ", " : "") + v[k];
    return s + "}";
}

/// AND, BIC, ORR, ORN (U = 0) and EOR, BSL, BIT, BIF (U = 1), selected by the size field.
string Logic(unsigned q, unsigned u, unsigned size, unsigned rd, unsigned rn, unsigned rm) {
    static const char* const ops[2][4] = {{"a & b", "a & ~b", "a | b", "a | ~b"},
                                          {"a ^ b", "(d & a) | (~d & b)", "(d & ~b) | (a & b)", "(d & b) | (a & ~b)"}};
    const string U = T(false, 64, 16);
    string b = Load(U, "a", rn) + Load(U, "b", rm);
    if (u && size) b += Load(U, "d", rd);
    return b + U + " r = " + ops[u][size] + "; " + Store(rd, "r", q);
}

/// AdvSIMD copy: DUP (element, general), INS (general, element), SMOV, UMOV.
string Copy(unsigned q, unsigned op, unsigned imm5, unsigned imm4, unsigned rd, unsigned rn) {
    const unsigned sz = imm5 & 1 ? 0 : imm5 & 2 ? 1 : imm5 & 4 ? 2 : imm5 & 8 ? 3 : 4;
    if (sz == 4) return "";
    const unsigned eb = 1u << sz, e = 8 * eb, idx = imm5 >> (sz + 1);
    const string ut = "uint" + to_string(e) + "_t", V = T(false, e, 16);
    auto get = [&](const string& type, unsigned r, unsigned k) {
        return type + " v; memcpy(&v, (const uint8_t*)" + Reg(r) + " + " + to_string(k * eb) + ", " + to_string(eb) +
               "); ";
    };
    auto put = [&](unsigned r, unsigned k) {
        return "memcpy((uint8_t*)" + Reg(r) + " + " + to_string(k * eb) + ", &v, " + to_string(eb) + "); ";
    };
    if (op) { // INS (element): Vd[idx] = Vn[imm4 >> size]
        if (!q) return "";
        return get(ut, rn, imm4 >> sz) + put(rd, idx);
    }
    switch (imm4) {
    case 0: // DUP (element)
        if (sz == 3 && !q) return "";
        return get(ut, rn, idx) + V + " r = (" + V + "){0} + v; " + Store(rd, "r", q);
    case 1: // DUP (general)
        if (sz == 3 && !q) return "";
        return ut + " v = (" + ut + ")" + Gpr(rn) + "; " + V + " r = (" + V + "){0} + v; " + Store(rd, "r", q);
    case 3: // INS (general)
        if (!q) return "";
        return ut + " v = (" + ut + ")" + Gpr(rn) + "; " + put(rd, idx);
    case 5: { // SMOV: to Wd (Q = 0; B, H) or Xd (Q = 1; B, H, S)
        if (sz == 3 || (sz == 2 && !q)) return "";
        if (rd == 31) return "(void)0; ";
        const string st = "int" + to_string(e) + "_t";
        return get(st, rn, idx) + "c->x[" + to_string(rd) + "] = " +
               (q ? string("(uint64_t)(int64_t)v; ") : string("(uint64_t)(uint32_t)(int32_t)v; "));
    }
    case 7: // UMOV: to Wd (Q = 0; B, H, S) or Xd (Q = 1; D)
        if (q ? sz != 3 : sz == 3) return "";
        if (rd == 31) return "(void)0; ";
        return get(ut, rn, idx) + "c->x[" + to_string(rd) + "] = (uint64_t)v; ";
    default: return "";
    }
}

/// EXT: bytes imm4 .. of Vm:Vn (Q = 0: of the low halves, imm4 < 8).
string Ext(unsigned q, unsigned imm4, unsigned rd, unsigned rn, unsigned rm) {
    if (!q && imm4 >= 8) return "";
    const unsigned n = q ? 16 : 8;
    std::vector<string> v(16, "0");
    for (unsigned k = 0; k < n; ++k) v[k] = k + imm4 < n ? Lane("a", k + imm4) : Lane("b", k + imm4 - n);
    const string B = T(false, 8, 16);
    return Load(B, "a", rn) + Load(B, "b", rm) + B + " r = " + Init(B, v) + "; " + Store(rd, "r", q);
}

/// UZP1/2, TRN1/2, ZIP1/2 (opcode 1, 5, 2, 6, 3, 7).
string Permute(unsigned q, unsigned size, unsigned op, unsigned rd, unsigned rn, unsigned rm) {
    if ((size == 3 && !q) || (op & 3) == 0) return "";
    const unsigned e = 8u << size, total = 128 / e, l = (q ? 64 : 32) * 2 / e; // l: lanes in use
    const bool two = op & 4;
    std::vector<string> v(total, "0");
    auto ab = [&](unsigned j) { return j < l ? Lane("a", j) : Lane("b", j - l); };
    for (unsigned k = 0; k < l; ++k) {
        switch (op & 3) {
        case 1: v[k] = ab(2 * k + two); break;                                        // UZP
        case 2: v[k] = Lane(k & 1 ? "b" : "a", (k & ~1u) + two); break;              // TRN
        case 3: v[k] = Lane(k & 1 ? "b" : "a", (two ? l / 2 : 0) + k / 2); break;      // ZIP
        }
    }
    const string V = T(false, e, 16);
    return Load(V, "a", rn) + Load(V, "b", rm) + V + " r = " + Init(V, v) + "; " + Store(rd, "r", q);
}

/// REV16 / REV32 / REV64: reverse e-bit elements within each container of `cbits`.
string Rev(unsigned q, unsigned size, unsigned cbits, unsigned rd, unsigned rn) {
    const unsigned eb = 1u << size, cb = cbits / 8, n = q ? 16 : 8;
    if (eb >= cb) return "";
    std::vector<string> v(16, "0");
    for (unsigned k = 0; k < n; ++k) {
        const unsigned base = k / cb * cb, in = k % cb;
        v[k] = Lane("a", base + (cb - eb - in / eb * eb) + in % eb);
    }
    const string B = T(false, 8, 16);
    return Load(B, "a", rn) + B + " r = " + Init(B, v) + "; " + Store(rd, "r", q);
}

/// VFPExpandImm: the 8-bit FP immediate as a single (dbl = false) or double.
uint64_t FpImm(unsigned imm8, bool dbl) {
    const uint64_t a = imm8 >> 7, b = (imm8 >> 6) & 1, rest = imm8 & 0x3f;
    if (!dbl) return (a << 31) | ((b ^ 1) << 30) | ((b ? 0x1fULL : 0) << 25) | (rest << 19);
    return (a << 63) | ((b ^ 1) << 62) | ((b ? 0xffULL : 0) << 54) | (rest << 48);
}

/// AdvSIMD modified immediate: MOVI, MVNI, ORR, BIC (immediate) and FMOV (vector, immediate).
string ModImm(unsigned q, unsigned op, unsigned cmode, unsigned imm8, unsigned rd) {
    const uint64_t i8 = imm8;
    auto rep32 = [](uint64_t v) { return v | v << 32; };
    auto rep16 = [&](uint64_t v) { return rep32(v | v << 16); };
    uint64_t imm;
    switch (cmode >> 1) {
    case 0: case 1: case 2: case 3: imm = rep32(i8 << (8 * (cmode >> 1))); break;
    case 4: case 5: imm = rep16(i8 << (8 * ((cmode >> 1) & 1))); break;
    case 6: imm = rep32(cmode & 1 ? (i8 << 16) | 0xffff : (i8 << 8) | 0xff); break;
    default:
        if (!(cmode & 1)) {
            if (!op) {
                imm = rep16(i8 | i8 << 8);
            } else { // MOVI (64-bit): each bit of imm8 a byte of ones or zeros
                imm = 0;
                for (unsigned k = 0; k < 8; ++k) imm |= (i8 >> k & 1 ? 0xffULL : 0) << (8 * k);
            }
        } else if (!op) {
            imm = rep32(FpImm(imm8, false)); // FMOV .2s / .4s
        } else {
            if (!q) return "";
            imm = FpImm(imm8, true); // FMOV .2d
        }
    }
    const string U = T(false, 64, 16), k = to_string(imm) + "ULL";
    const bool orr_bic = cmode < 12 && (cmode & 1);
    if (orr_bic) {
        return Load(U, "d", rd) + U + " r = d " + (op ? "& ~" : "| ") + "(" + U + "){" + k + ", " + k + "}; " +
               Store(rd, "r", q);
    }
    const bool mvn = op && cmode < 14;
    return U + " r = " + (mvn ? "~" : "") + "(" + U + "){" + k + ", " + k + "}; " + Store(rd, "r", q);
}

/// The condition of FCSEL & co. from the flags (c->n, c->z, c->c, c->v).
string Cond(unsigned cond) {
    static const char* const base[8] = {"c->z", "c->c", "c->n", "c->v", "(c->c && !c->z)", "(c->n == c->v)",
                                        "(!c->z && c->n == c->v)", "1"};
    const string b = base[cond >> 1];
    return (cond & 1) && cond != 15 ? "!" + b : b;
}

/// Scalar FP moves that are exact by nature (no FPSR bit, no rounding, FPCR ignored): FMOV / FABS / FNEG
/// (register), FMOV (immediate), FCSEL, FMOV (general). S and D forms; the result zeroes the rest of Vd.
string ScalarFp(uint32_t i) {
    const unsigned rd = i & 31, rn = (i >> 5) & 31, rm = (i >> 16) & 31, ty = (i >> 22) & 3;
    const string d0 = Reg(rd) + "[0]", d1 = Reg(rd) + "[1]";
    const string w = ty ? "" : " & 0xFFFFFFFFULL";
    if ((i & 0xFF207C00u) == 0x1E204000u) { // 0 0 0 11110 ty 1 opcode(6) 10000 Rn Rd
        const unsigned opc = (i >> 15) & 0x3f;
        if (ty > 1 || opc > 2) return "";
        const string sign = ty ? "0x8000000000000000ULL" : "0x80000000ULL";
        const string v = opc == 0 ? "a" : opc == 1 ? "a & ~" + sign : "a ^ " + sign;
        return "uint64_t a = " + Reg(rn) + "[0]" + w + "; " + d0 + " = " + v + "; " + d1 + " = 0; ";
    }
    if ((i & 0xFF201FE0u) == 0x1E201000u) { // FMOV (scalar, immediate): 0 0 0 11110 ty 1 imm8 100 00000 Rd
        if (ty > 1) return "";
        return d0 + " = " + to_string(FpImm((i >> 13) & 0xff, ty)) + "ULL; " + d1 + " = 0; ";
    }
    if ((i & 0xFF200C00u) == 0x1E200C00u) { // FCSEL: 0 0 0 11110 ty 1 Rm cond 11 Rn Rd
        if (ty > 1) return "";
        return "uint64_t r = " + Cond((i >> 12) & 15) + " ? " + Reg(rn) + "[0] : " + Reg(rm) + "[0]; " + d0 + " = r" +
               w + "; " + d1 + " = 0; ";
    }
    if ((i & 0x7F20FC00u) == 0x1E200000u) { // sf 0 0 11110 ty 1 rmode opcode 000000 Rn Rd: FMOV (general)
        const unsigned sf = i >> 31, rmode = (i >> 19) & 3, opc = (i >> 16) & 7;
        const unsigned kind = !sf && ty == 0 && rmode == 0 ? 1 : sf && ty == 1 && rmode == 0 ? 2
                            : sf && ty == 2 && rmode == 1 ? 3 : 0; // S <-> W, D <-> X, V.D[1] <-> X
        if (!kind || (opc != 6 && opc != 7)) return "";
        if (opc == 6) { // to the general register
            if (rd == 31) return "(void)0; ";
            const string v = kind == 3 ? Reg(rn) + "[1]" : Reg(rn) + "[0]" + (kind == 1 ? " & 0xFFFFFFFFULL" : "");
            return "c->x[" + to_string(rd) + "] = " + v + "; ";
        }
        const string v = Gpr(rn) + (kind == 1 ? " & 0xFFFFFFFFULL" : "");
        if (kind == 3) return d1 + " = " + v + "; ";
        return "uint64_t v = " + v + "; " + d0 + " = v; " + d1 + " = 0; ";
    }
    return "";
}

// --- shift by immediate: 0 Q U 011110 immh immb opcode 1 Rn Rd (immh != 0) -------------------------
string ShiftImm(unsigned q, unsigned u, unsigned immh, unsigned immb, unsigned op, unsigned rd, unsigned rn) {
    const unsigned e = immh & 8 ? 64 : immh & 4 ? 32 : immh & 2 ? 16 : 8;
    const unsigned ih = (immh << 3) | immb, sr = 2 * e - ih, sl = ih - e;
    const bool s = !u;
    switch (op) {
    case 0x00: case 0x02: case 0x04: case 0x06: { // [S|U]SHR, [S|U]SRA, [S|U]RSHR, [S|U]RSRA
        if (e == 64 && !q) return "";
        const string A = T(s, e, 16), U = T(false, e, 16);
        string b = Load(A, "a", rn);
        const string v = op & 4 ? RoundShr("a", A, s, e, sr) : Shr("a", A, s, e, sr);
        if (op & 2) b += Load(U, "d", rd) + U + " r = d + (" + U + ")" + v + "; ";
        else b += U + " r = (" + U + ")" + v + "; ";
        return b + Store(rd, "r", q);
    }
    case 0x0a: { // SHL (SLI: U = 1)
        if (u || (e == 64 && !q)) return "";
        const string U = T(false, e, 16);
        return Load(U, "a", rn) + U + " r = a << " + to_string(sl) + "; " + Store(rd, "r", q);
    }
    case 0x14: { // SSHLL / USHLL (2): widen the low (Q = 0) or high (Q = 1) half, shift left by 0 .. e-1
        if (e == 64) return "";
        const string N = T(s, e, 8), W = T(s, 2 * e, 16), WU = T(false, 2 * e, 16);
        return Load(N, "n", rn, q ? 8 : 0) + WU + " r = (" + WU + ")__builtin_convertvector(n, " + W + ") << " +
               to_string(sl) + "; " + Store(rd, "r", 1);
    }
    case 0x10: case 0x11: case 0x12: case 0x13: { // narrowing: the destination lanes are e bits
        if (e == 64) return "";
        const bool round = op & 1;
        // SHRN / RSHRN (U = 0, op 0x10/0x11): no saturation; SQSHRUN / SQRSHRUN (U = 1, 0x10/0x11): signed ->
        // unsigned; SQSHRN / SQRSHRN (U = 0, 0x12/0x13): signed -> signed; UQSHRN / UQRSHRN (U = 1): unsigned.
        const bool plain = op <= 0x11 && !u;
        const bool src_s = !(op >= 0x12 && u);
        const bool dst_s = op >= 0x12 && !u;
        const string W = T(plain ? false : src_s, 2 * e, 16);
        string b = Load(W, "a", rn);
        const string v = round ? RoundShr("a", W, plain ? false : src_s, 2 * e, sr)
                               : Shr("a", W, plain ? false : src_s, 2 * e, sr);
        b += W + " t = " + v + "; ";
        if (plain) {
            b += T(false, e, 8) + " n = __builtin_convertvector(t, " + T(false, e, 8) + "); ";
            return b + StoreNarrow(rd, "n", q);
        }
        b += Clamp("t", W, src_s, 2 * e, Min(dst_s, e), Max(dst_s, e), src_s, src_s);
        b += T(dst_s, e, 8) + " n = __builtin_convertvector(r, " + T(dst_s, e, 8) + "); ";
        return b + Qc("m", 16) + StoreNarrow(rd, "n", q);
    }
    default: return "";
    }
}

} // namespace

bool Emit(uint32_t i, std::string& out) {
    std::string body = ScalarFp(i);
    if (!body.empty()) {
        out += "{ " + body + "}\n";
        return true;
    }
    if (i >> 31) return false; // Advanced SIMD has bit 31 clear; the scalar forms differ in bit 28 (not matched below)
    const unsigned q = (i >> 30) & 1, u = (i >> 29) & 1;
    const unsigned rd = i & 31, rn = (i >> 5) & 31, rm = (i >> 16) & 31;
    if ((i & 0x9FE08400u) == 0x0E000400u) { // copy: 0 Q op 01110000 imm5 0 imm4 1 Rn Rd
        body = Copy(q, u, (i >> 16) & 31, (i >> 11) & 15, rd, rn);
    } else if ((i & 0xFFE0FC00u) == 0x5E000400u) { // DUP (element, scalar): 01 0 11110000 imm5 0 0000 1 Rn Rd
        const unsigned imm5 = (i >> 16) & 31, sz = imm5 & 1 ? 0 : imm5 & 2 ? 1 : imm5 & 4 ? 2 : imm5 & 8 ? 3 : 4;
        if (sz < 4) {
            const unsigned eb = 1u << sz;
            body = "uint64_t v = 0; memcpy(&v, (const uint8_t*)" + Reg(rn) + " + " + to_string((imm5 >> (sz + 1)) * eb) +
                   ", " + to_string(eb) + "); " + Reg(rd) + "[0] = v; " + Reg(rd) + "[1] = 0; ";
        }
    } else if ((i & 0xBF208C00u) == 0x0E000800u) { // permute: 0 Q 001110 size 0 Rm 0 opcode 10 Rn Rd
        body = Permute(q, (i >> 22) & 3, (i >> 12) & 7, rd, rn, rm);
    } else if ((i & 0xBFE08400u) == 0x2E000000u) { // EXT: 0 Q 101110 00 0 Rm 0 imm4 0 Rn Rd
        body = Ext(q, (i >> 11) & 15, rd, rn, rm);
    } else if ((i & 0x9FF80C00u) == 0x0F000400u) { // modified immediate (o2 = 0): 0 Q op 0111100000 abc cmode 0 1 defgh Rd
        body = ModImm(q, u, (i >> 12) & 15, (((i >> 16) & 7) << 5) | ((i >> 5) & 31), rd);
    } else if (((i >> 24) & 0x1f) == 0x0e && ((i >> 21) & 1)) { // 0 Q U 01110 xx 1 ...
        const unsigned size = (i >> 22) & 3;
        if ((i >> 10) & 1) {
            const unsigned op = (i >> 11) & 0x1f;
            body = op == 0x03 ? Logic(q, u, size, rd, rn, rm) : ThreeSame(q, u, size, op, rd, rn, rm);
        } else if (((i >> 10) & 3) == 0) {
            body = ThreeDiff(q, u, size, (i >> 12) & 0xf, rd, rn, rm);
        } else if (((i >> 17) & 0x1f) == 0x10 && ((i >> 10) & 3) == 2) {
            body = TwoMisc(q, u, size, (i >> 12) & 0x1f, rd, rn);
        }
    } else if (((i >> 23) & 0x3f) == 0x1e && ((i >> 10) & 1) && ((i >> 19) & 0xf)) { // 0 Q U 011110 immh != 0
        body = ShiftImm(q, u, (i >> 19) & 0xf, (i >> 16) & 7, (i >> 11) & 0x1f, rd, rn);
    }
    if (body.empty()) return false;
    out += "{ " + body + "}\n";
    return true;
}

// --- FP fast path -----------------------------------------------------------
// One call per site into nx_simd.h's nxv_fpx_* (constant arguments: the compiler specialises each), which
// returns 1 after storing a kept result and 0, changing nothing, when the exact translation must run.
bool EmitFp(uint32_t i, const std::string& fallback, std::string& out) {
    const unsigned rd = i & 31, rn = (i >> 5) & 31;
    std::string call;
    if ((i & 0xFF200C00u) == 0x1E200800u) { // scalar FP 2-source: 0 0 0 11110 ty 1 Rm opcode 10 Rn Rd
        const unsigned ty = (i >> 22) & 3, op = (i >> 12) & 15, rm = (i >> 16) & 31;
        // FMAXNM / FMINNM equal FMAX / FMIN when no operand is a NaN, and a NaN takes the exact path
        static const char* const ops[16] = {"NXV_FMUL", "NXV_FDIV", "NXV_FADD", "NXV_FSUB", "NXV_FMAX", "NXV_FMIN",
                                            "NXV_FMAX", "NXV_FMIN"};
        const char* k = ops[op];
        if (!k || ty > 1) return false;
        call = string(ty ? "nxv_fpx_s64" : "nxv_fpx_s32") + "(c, " + k + ", " + to_string(rd) + ", c->vreg[" +
               to_string(rn) + "][0], c->vreg[" + to_string(rm) + "][0])";
    } else if ((i & 0xFF20FC07u) == 0x1E202000u) { // FCMP / FCMPE: 0 0 0 11110 ty 1 Rm 001000 Rn opc 000
        const unsigned ty = (i >> 22) & 3, rm = (i >> 16) & 31;
        if (ty > 1) return false;
        call = string(ty ? "nxv_fpx_fcmp64" : "nxv_fpx_fcmp32") + "(c, c->vreg[" + to_string(rn) + "][0], " +
               ((i >> 3) & 1 ? string("0") : "c->vreg[" + to_string(rm) + "][0]") + ")";
    } else if ((i & 0x9F200400u) == 0x0E200400u) { // vector three same: 0 Q U 01110 a sz 1 Rm opcode 1 Rn Rd
        const unsigned q = (i >> 30) & 1, u = (i >> 29) & 1, a = (i >> 23) & 1, sz = (i >> 22) & 1,
                       op = (i >> 11) & 0x1f, rm = (i >> 16) & 31;
        if (op < 0x18) return false;
        // [opcode - 0x18][U][a]
        static const char* const ops[8][2][2] = {
            {{"NXV_FMAX", "NXV_FMIN"}, {nullptr, nullptr}},     // 11000 FMAXNM, FMINNM (as FMAX / FMIN)
            {{nullptr, nullptr}, {nullptr, nullptr}},           // 11001 FMLA, FMLS
            {{"NXV_FADD", "NXV_FSUB"}, {"NXV_FADDP", "NXV_FABD"}},
            {{nullptr, nullptr}, {"NXV_FMUL", nullptr}},        // 11011 FMULX, FMUL
            {{"NXV_FCMEQ", nullptr}, {"NXV_FCMGE", "NXV_FCMGT"}},
            {{nullptr, nullptr}, {"NXV_FACGE", "NXV_FACGT"}},
            {{"NXV_FMAX", "NXV_FMIN"}, {nullptr, nullptr}},     // 11110 FMAX, FMIN (FMAXP, FMINP)
            {{"NXV_FRECPS", "NXV_FRSQRTS"}, {"NXV_FDIV", nullptr}}};
        const char* k = ops[op - 0x18][u][a];
        if (!k || (sz && !q)) return false; // .1d is reserved
        if (sz && (u == 0 && op == 0x1f)) return false; // FRECPS / FRSQRTS .2d: need a binary64 FMA
        call = string(sz ? "nxv_fpx_v64" : "nxv_fpx_v32") + "(c, " + k + ", " + to_string(rd) + ", " + to_string(rn) +
               ", " + to_string(rm) + ", " + to_string(sz ? 2 : q ? 4 : 2) + ")";
    } else if ((i & 0x9FBE0C00u) == 0x0EA00800u) { // compare with zero: 0 Q U 01110 1 sz 10000 opcode 10 Rn Rd
        const unsigned q = (i >> 30) & 1, u = (i >> 29) & 1, sz = (i >> 22) & 1, op = (i >> 12) & 0x1f;
        // FCMGT / FCMGE #0: a > 0, a >= 0; FCMEQ #0; FCMLE / FCMLT #0: 0 >= a, 0 > a (operands swapped)
        const char* k = op == 0x0c ? (u ? "NXV_FCMGE" : "NXV_FCMGT") : op == 0x0d ? (u ? "NXV_FCMGE" : "NXV_FCMEQ")
                      : op == 0x0e && !u ? "NXV_FCMGT" : nullptr;
        if (!k || (sz && !q)) return false;
        const bool swap = (op == 0x0d && u) || op == 0x0e;
        call = string(sz ? "nxv_fpx_v64z" : "nxv_fpx_v32z") + "(c, " + k + ", " + to_string(rd) + ", " + to_string(rn) +
               ", " + (swap ? "1" : "0") + ", " + to_string(sz ? 2 : q ? 4 : 2) + ")";
    } else if ((i & 0xBF80F400u) == 0x0F809000u || (i & 0xFF80F400u) == 0x5F809000u) {
        // FMUL by element: 0 Q 0 01111 1 sz L M Rm 1001 H 0 Rn Rd (vector), 01 0 11111 1 sz ... (scalar)
        const bool scalar = (i >> 28) & 1;
        const unsigned q = (i >> 30) & 1, sz = (i >> 22) & 1, l = (i >> 21) & 1, h = (i >> 11) & 1,
                       rm = (i >> 16) & 31;
        if (sz && (l || (!scalar && !q))) return false; // reserved
        const unsigned idx = sz ? h : (h << 1) | l;
        if (scalar) {
            call = string(sz ? "nxv_fpx_s64" : "nxv_fpx_s32") + "(c, NXV_FMUL, " + to_string(rd) + ", c->vreg[" +
                   to_string(rn) + "][0], " +
                   (sz ? "c->vreg[" + to_string(rm) + "][" + to_string(idx) + "]"
                       : "nxv_fpx_lane32(c, " + to_string(rm) + ", " + to_string(idx) + ")") + ")";
        } else {
            call = string(sz ? "nxv_fpx_v64x" : "nxv_fpx_v32x") + "(c, NXV_FMUL, " + to_string(rd) + ", " +
                   to_string(rn) + ", " + to_string(rm) + ", " + to_string(idx) + ", " + to_string(sz ? 2 : q ? 4 : 2) + ")";
        }
    } else {
        return false;
    }
    out += "{ if (!" + call + ") { " + fallback + " } }\n";
    return true;
}

const std::string& Header() {
    static const std::string h = R"(/* nx_simd.h: written by nxrecomp (lifter/src/simd.cpp). */
#pragma once
#include <stdint.h>
#include <string.h>

/* 64-, 128- and 256-bit vectors; nxv_<s|u><lane bits>x<lanes>. */
#define NXV_T(name, lane, bytes) typedef lane name __attribute__((vector_size(bytes)))
NXV_T(nxv_s8x8, int8_t, 8);    NXV_T(nxv_u8x8, uint8_t, 8);
NXV_T(nxv_s8x16, int8_t, 16);  NXV_T(nxv_u8x16, uint8_t, 16);
NXV_T(nxv_s16x4, int16_t, 8);  NXV_T(nxv_u16x4, uint16_t, 8);
NXV_T(nxv_s16x8, int16_t, 16); NXV_T(nxv_u16x8, uint16_t, 16);
NXV_T(nxv_s32x2, int32_t, 8);  NXV_T(nxv_u32x2, uint32_t, 8);
NXV_T(nxv_s32x4, int32_t, 16); NXV_T(nxv_u32x4, uint32_t, 16);
NXV_T(nxv_s64x1, int64_t, 8);  NXV_T(nxv_u64x1, uint64_t, 8);
NXV_T(nxv_s64x2, int64_t, 16); NXV_T(nxv_u64x2, uint64_t, 16);
NXV_T(nxv_s16x16, int16_t, 32); NXV_T(nxv_u16x16, uint16_t, 32);
NXV_T(nxv_s32x8, int32_t, 32);  NXV_T(nxv_u32x8, uint32_t, 32);
NXV_T(nxv_s64x4, int64_t, 32);  NXV_T(nxv_u64x4, uint64_t, 32);
#undef NXV_T

/* FP fast path (lifter/src/simd.cpp EmitFp): drippu's FPX conditions on all lanes at
   once. Arithmetic (FADD, FSUB, FMUL, FADDP, FDIV, FABD, FRECPS, FRSQRTS) stores a result only when the gate
   RECOMP_FPX_OPEN holds (FPSR.IXC set, FPCR default) and every lane is kept as drippu's per-lane fast path would
   keep it (sums: finite; products and quotients: larger than the smallest normal, or a finite one of a zero;
   FRECPS/FRSQRTS: computed in binary64 from the exact product, larger than the smallest normal and not on a
   binary32 midpoint). Compares, FCMP and FMAX/FMIN set no FPSR bit and round nothing when no operand is a NaN
   and FPCR is the default (FZ clear: no input is flushed), so they need only NXV_FPX_QUIET and no NaN lane; FMAX
   of equal operands is their AND (+0 for +0, -0), FMIN their OR. The functions then return 1. Otherwise nothing
   changes and they return 0: the caller runs the exact translation. The host FP mode is FPX's contract (MXCSR
   round to nearest, DAZ/FTZ clear). NXV_FPX_MUTATE is a difftest hook (--mutate-fpx flips the low bit of each
   kept result, FCMP's V flag). */
#ifdef RECOMP_FPX_FPCR_MASK /* drippu's FPX macros (absent when the translation has FPX off) */
#ifndef NXV_FPX_MUTATE
#define NXV_FPX_MUTATE
#endif
enum { NXV_FADD, NXV_FSUB, NXV_FMUL, NXV_FADDP, NXV_FDIV, NXV_FABD, NXV_FRECPS, NXV_FRSQRTS,
       NXV_FMAX, NXV_FMIN, NXV_FCMEQ, NXV_FCMGE, NXV_FCMGT, NXV_FACGE, NXV_FACGT };
#define NXV_FPX_QUIET_OP(op) ((op) >= NXV_FMAX)
#define NXV_FPX_QUIET(c) (RECOMP_FPX_HOST && !((c)->fpcr & RECOMP_FPX_FPCR_MASK))
#define NXV_FPX_GATE(c, op) (NXV_FPX_QUIET_OP(op) ? NXV_FPX_QUIET(c) : (RECOMP_FPX_HOST && RECOMP_FPX_OPEN(c)))
#define NXV_NAN32(x) (((x) & 0x7fffffffu) > 0x7f800000u)
#define NXV_NAN64(x) (((x) & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL)
typedef float nxv_f32x4 __attribute__((vector_size(16)));
typedef double nxv_f64x2 __attribute__((vector_size(16)));
typedef double nxv_f64x4 __attribute__((vector_size(32)));

static inline uint64_t nxv_fpx_lane32(const GuestContext* c, unsigned r, unsigned i) {
    uint32_t v;
    memcpy(&v, (const uint8_t*)c->vreg[r] + 4 * i, 4);
    return v;
}
static inline int nxv_fpx_s32(GuestContext* c, int op, unsigned d, uint64_t a, uint64_t b) {
    if (!NXV_FPX_GATE(c, op)) return 0;
    a = (uint32_t)a, b = (uint32_t)b;
    const float x = recomp_fpx_f(a), y = recomp_fpx_f(b);
    uint64_t t;
    if (op == NXV_FMAX || op == NXV_FMIN) {
        if (NXV_NAN32(a) || NXV_NAN32(b)) return 0;
        t = x > y ? (op == NXV_FMAX ? a : b) : x < y ? (op == NXV_FMAX ? b : a) : op == NXV_FMAX ? (a & b) : (a | b);
    } else {
        t = recomp_fpx_fb(op == NXV_FADD ? x + y : op == NXV_FSUB ? x - y : op == NXV_FDIV ? x / y : x * y);
        if (!(op == NXV_FMUL ? RECOMP_FPX_KEEP32(t, a, b) : op == NXV_FDIV ? RECOMP_FPX_KEEPDIV32(t, a)
                                                                             : RECOMP_FPX_FIN32(t))) return 0;
    }
    t = (uint32_t)(t NXV_FPX_MUTATE);
    c->vreg[d][0] = t, c->vreg[d][1] = 0;
    return 1;
}
static inline int nxv_fpx_s64(GuestContext* c, int op, unsigned d, uint64_t a, uint64_t b) {
    if (!NXV_FPX_GATE(c, op)) return 0;
    const double x = recomp_fpx_d(a), y = recomp_fpx_d(b);
    uint64_t t;
    if (op == NXV_FMAX || op == NXV_FMIN) {
        if (NXV_NAN64(a) || NXV_NAN64(b)) return 0;
        t = x > y ? (op == NXV_FMAX ? a : b) : x < y ? (op == NXV_FMAX ? b : a) : op == NXV_FMAX ? (a & b) : (a | b);
    } else {
        t = recomp_fpx_db(op == NXV_FADD ? x + y : op == NXV_FSUB ? x - y : op == NXV_FDIV ? x / y : x * y);
        if (!(op == NXV_FMUL ? RECOMP_FPX_KEEP64(t, a, b) : op == NXV_FDIV ? RECOMP_FPX_KEEPDIV64(t, a)
                                                                             : RECOMP_FPX_FIN64(t))) return 0;
    }
    t = t NXV_FPX_MUTATE;
    c->vreg[d][0] = t, c->vreg[d][1] = 0;
    return 1;
}
/* FCMP / FCMPE (b = 0 for the #0.0 forms): NZCV = 0110 equal, 1000 less, 0010 greater. */
static inline int nxv_fpx_fcmp32(GuestContext* c, uint64_t a, uint64_t b) {
    a = (uint32_t)a, b = (uint32_t)b;
    if (!NXV_FPX_QUIET(c) || NXV_NAN32(a) || NXV_NAN32(b)) return 0;
    const float x = recomp_fpx_f(a), y = recomp_fpx_f(b);
    unsigned f = x == y ? 6u : x < y ? 8u : 2u;
    f = (unsigned)(f NXV_FPX_MUTATE);
    c->n = (uint8_t)(f >> 3 & 1), c->z = (uint8_t)(f >> 2 & 1), c->c = (uint8_t)(f >> 1 & 1), c->v = (uint8_t)(f & 1);
    return 1;
}
static inline int nxv_fpx_fcmp64(GuestContext* c, uint64_t a, uint64_t b) {
    if (!NXV_FPX_QUIET(c) || NXV_NAN64(a) || NXV_NAN64(b)) return 0;
    const double x = recomp_fpx_d(a), y = recomp_fpx_d(b);
    unsigned f = x == y ? 6u : x < y ? 8u : 2u;
    f = (unsigned)(f NXV_FPX_MUTATE);
    c->n = (uint8_t)(f >> 3 & 1), c->z = (uint8_t)(f >> 2 & 1), c->c = (uint8_t)(f >> 1 & 1), c->v = (uint8_t)(f & 1);
    return 1;
}
/* 32-bit lanes: `lanes` 2 (the low half; the high half of Vd is zeroed) or 4. */
static inline int nxv_fpx_core32(GuestContext* c, int op, unsigned d, nxv_f32x4 a, nxv_f32x4 b, unsigned lanes) {
    nxv_u32x4 tb, ab, bb;
    nxv_s32x4 keep;
    memcpy(&ab, &a, 16), memcpy(&bb, &b, 16);
    if (NXV_FPX_QUIET_OP(op)) {
        keep = (nxv_s32x4)(((ab & 0x7fffffffu) <= 0x7f800000u) & ((bb & 0x7fffffffu) <= 0x7f800000u));
        if (op == NXV_FMAX || op == NXV_FMIN) {
            const nxv_s32x4 gt = (nxv_s32x4)(a > b), lt = (nxv_s32x4)(a < b);
            const nxv_u32x4 hi = op == NXV_FMAX ? ab : bb, lo = op == NXV_FMAX ? bb : ab;
            const nxv_u32x4 eq = op == NXV_FMAX ? ab & bb : ab | bb;
            tb = (hi & (nxv_u32x4)gt) | (lo & (nxv_u32x4)lt) | (eq & ~(nxv_u32x4)(gt | lt));
        } else {
            nxv_f32x4 x = a, y = b;
            if (op == NXV_FACGE || op == NXV_FACGT) {
                const nxv_u32x4 xa = ab & 0x7fffffffu, ya = bb & 0x7fffffffu;
                memcpy(&x, &xa, 16), memcpy(&y, &ya, 16);
            }
            tb = op == NXV_FCMEQ ? (nxv_u32x4)(x == y) : op == NXV_FCMGE || op == NXV_FACGE ? (nxv_u32x4)(x >= y)
                                                                                   : (nxv_u32x4)(x > y);
        }
    } else {
        nxv_f32x4 t;
        nxv_s32x4 mid = {0, 0, 0, 0};
        if (op == NXV_FADDP) { /* pairs of Vn then of Vm: [n0+n1, n2+n3, m0+m1, m2+m3], .2s [n0+n1, m0+m1] */
            const nxv_f32x4 x = lanes == 4 ? (nxv_f32x4){a[0], a[2], b[0], b[2]} : (nxv_f32x4){a[0], b[0], 0, 0};
            const nxv_f32x4 y = lanes == 4 ? (nxv_f32x4){a[1], a[3], b[1], b[3]} : (nxv_f32x4){a[1], b[1], 0, 0};
            t = x + y;
        } else if (op == NXV_FRECPS || op == NXV_FRSQRTS) {
            /* binary64: the product is exact, the difference rounded once; refused on a binary32 midpoint */
            const nxv_f64x4 x = __builtin_convertvector(a, nxv_f64x4), y = __builtin_convertvector(b, nxv_f64x4);
            const nxv_f64x4 s = op == NXV_FRECPS ? 2.0 - x * y : (3.0 - x * y) * 0.5;
            nxv_u64x4 sb;
            memcpy(&sb, &s, 32);
            mid = __builtin_convertvector((sb & 0x1fffffffULL) == 0x10000000ULL, nxv_s32x4);
            t = __builtin_convertvector(s, nxv_f32x4);
        } else {
            t = op == NXV_FADD ? a + b : op == NXV_FSUB || op == NXV_FABD ? a - b : op == NXV_FDIV ? a / b : a * b;
        }
        memcpy(&tb, &t, 16);
        const nxv_s32x4 fin = (nxv_s32x4)((tb & 0x7f800000u) != 0x7f800000u);
        const nxv_s32x4 big = (nxv_s32x4)(((tb & 0x7fffffffu) - 0x00800001u) < 0x7effffffu);
        if (op == NXV_FMUL) {
            keep = big | (((nxv_s32x4)((ab & 0x7fffffffu) == 0) | (nxv_s32x4)((bb & 0x7fffffffu) == 0)) & fin);
        } else if (op == NXV_FDIV) {
            keep = big | ((nxv_s32x4)((ab & 0x7fffffffu) == 0) & fin);
        } else if (op == NXV_FRECPS || op == NXV_FRSQRTS) {
            keep = big & ~mid;
        } else {
            keep = fin;
        }
        if (op == NXV_FABD) tb &= 0x7fffffffu;
    }
    uint64_t k[2];
    memcpy(k, &keep, 16);
    if (lanes == 4 ? (k[0] & k[1]) != ~0ULL : k[0] != ~0ULL) return 0;
    tb = tb NXV_FPX_MUTATE;
    if (lanes == 4) memcpy(c->vreg[d], &tb, 16);
    else memcpy(c->vreg[d], &tb, 8), c->vreg[d][1] = 0;
    return 1;
}
static inline int nxv_fpx_v32(GuestContext* c, int op, unsigned d, unsigned n, unsigned m, unsigned lanes) {
    if (!NXV_FPX_GATE(c, op)) return 0;
    nxv_f32x4 a, b;
    memcpy(&a, c->vreg[n], 16), memcpy(&b, c->vreg[m], 16);
    return nxv_fpx_core32(c, op, d, a, b, lanes);
}
static inline int nxv_fpx_v32x(GuestContext* c, int op, unsigned d, unsigned n, unsigned m, unsigned idx, unsigned lanes) {
    if (!NXV_FPX_GATE(c, op)) return 0;
    nxv_f32x4 a;
    memcpy(&a, c->vreg[n], 16);
    const float e = recomp_fpx_f(nxv_fpx_lane32(c, m, idx));
    return nxv_fpx_core32(c, op, d, a, (nxv_f32x4){e, e, e, e}, lanes);
}
/* Compares with zero; `swap`: zero is the first operand (FCMLE, FCMLT). */
static inline int nxv_fpx_v32z(GuestContext* c, int op, unsigned d, unsigned n, int swap, unsigned lanes) {
    if (!NXV_FPX_GATE(c, op)) return 0;
    nxv_f32x4 a;
    memcpy(&a, c->vreg[n], 16);
    const nxv_f32x4 z = {0, 0, 0, 0};
    return swap ? nxv_fpx_core32(c, op, d, z, a, lanes) : nxv_fpx_core32(c, op, d, a, z, lanes);
}
static inline int nxv_fpx_core64(GuestContext* c, int op, unsigned d, nxv_f64x2 a, nxv_f64x2 b) {
    nxv_u64x2 tb, ab, bb;
    nxv_s64x2 keep;
    memcpy(&ab, &a, 16), memcpy(&bb, &b, 16);
    if (NXV_FPX_QUIET_OP(op)) {
        keep = (nxv_s64x2)(((ab & 0x7fffffffffffffffULL) <= 0x7ff0000000000000ULL) &
                           ((bb & 0x7fffffffffffffffULL) <= 0x7ff0000000000000ULL));
        if (op == NXV_FMAX || op == NXV_FMIN) {
            const nxv_s64x2 gt = (nxv_s64x2)(a > b), lt = (nxv_s64x2)(a < b);
            const nxv_u64x2 hi = op == NXV_FMAX ? ab : bb, lo = op == NXV_FMAX ? bb : ab;
            const nxv_u64x2 eq = op == NXV_FMAX ? ab & bb : ab | bb;
            tb = (hi & (nxv_u64x2)gt) | (lo & (nxv_u64x2)lt) | (eq & ~(nxv_u64x2)(gt | lt));
        } else {
            nxv_f64x2 x = a, y = b;
            if (op == NXV_FACGE || op == NXV_FACGT) {
                const nxv_u64x2 xa = ab & 0x7fffffffffffffffULL, ya = bb & 0x7fffffffffffffffULL;
                memcpy(&x, &xa, 16), memcpy(&y, &ya, 16);
            }
            tb = op == NXV_FCMEQ ? (nxv_u64x2)(x == y) : op == NXV_FCMGE || op == NXV_FACGE ? (nxv_u64x2)(x >= y)
                                                                                   : (nxv_u64x2)(x > y);
        }
    } else {
        const nxv_f64x2 t = op == NXV_FADDP ? (nxv_f64x2){a[0], b[0]} + (nxv_f64x2){a[1], b[1]}
                          : op == NXV_FADD  ? a + b : op == NXV_FSUB || op == NXV_FABD ? a - b
                          : op == NXV_FDIV  ? a / b : a * b;
        memcpy(&tb, &t, 16);
        const nxv_s64x2 fin = (nxv_s64x2)((tb & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL);
        const nxv_s64x2 big =
            (nxv_s64x2)(((tb & 0x7fffffffffffffffULL) - 0x0010000000000001ULL) < 0x7fdfffffffffffffULL);
        if (op == NXV_FMUL) {
            keep = big | (((nxv_s64x2)((ab & 0x7fffffffffffffffULL) == 0) |
                           (nxv_s64x2)((bb & 0x7fffffffffffffffULL) == 0)) & fin);
        } else if (op == NXV_FDIV) {
            keep = big | ((nxv_s64x2)((ab & 0x7fffffffffffffffULL) == 0) & fin);
        } else {
            keep = fin;
        }
        if (op == NXV_FABD) tb &= 0x7fffffffffffffffULL;
    }
    if ((keep[0] & keep[1]) != -1) return 0;
    tb = tb NXV_FPX_MUTATE;
    memcpy(c->vreg[d], &tb, 16);
    return 1;
}
static inline int nxv_fpx_v64(GuestContext* c, int op, unsigned d, unsigned n, unsigned m, unsigned lanes) {
    if (!NXV_FPX_GATE(c, op)) return 0;
    nxv_f64x2 a, b;
    memcpy(&a, c->vreg[n], 16), memcpy(&b, c->vreg[m], 16);
    (void)lanes;
    return nxv_fpx_core64(c, op, d, a, b);
}
static inline int nxv_fpx_v64x(GuestContext* c, int op, unsigned d, unsigned n, unsigned m, unsigned idx, unsigned lanes) {
    if (!NXV_FPX_GATE(c, op)) return 0;
    nxv_f64x2 a;
    memcpy(&a, c->vreg[n], 16);
    const double e = recomp_fpx_d(c->vreg[m][idx]);
    (void)lanes;
    return nxv_fpx_core64(c, op, d, a, (nxv_f64x2){e, e});
}
static inline int nxv_fpx_v64z(GuestContext* c, int op, unsigned d, unsigned n, int swap, unsigned lanes) {
    if (!NXV_FPX_GATE(c, op)) return 0;
    nxv_f64x2 a;
    memcpy(&a, c->vreg[n], 16);
    const nxv_f64x2 z = {0, 0};
    (void)lanes;
    return swap ? nxv_fpx_core64(c, op, d, z, a) : nxv_fpx_core64(c, op, d, a, z);
}

#endif /* RECOMP_FPX_FPCR_MASK */

/* FPSR.QC (bit 27) when any lane of the first `bytes` (8 or 16) of a mask is set; branch-free. */
static inline uint64_t nxv_qc(const void* mask, size_t bytes) {
    uint64_t q[2] = {0, 0};
    memcpy(q, mask, bytes);
    return (uint64_t)((q[0] | q[1]) != 0) << 27;
}
)";
    return h;
}

} // namespace nxrecomp::simd
