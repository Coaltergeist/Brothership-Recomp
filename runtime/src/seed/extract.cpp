// The game's shader stages from its own files (extract.h).
#include "extract.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "oodle.h"

namespace fs = std::filesystem;

namespace rt::seed {
namespace {

using Bytes = std::vector<uint8_t>;

// --- SHA-1 (FIPS 180-4): the stages' names, as shaderseed.py gives them -------------------------------------------

struct Sha1 {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    uint8_t block[64];
    size_t used = 0;
    uint64_t total = 0;

    static uint32_t Rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
    void Compress() {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i) {
            w[i] = uint32_t{block[4 * i]} << 24 | uint32_t{block[4 * i + 1]} << 16 | uint32_t{block[4 * i + 2]} << 8 |
                   block[4 * i + 3];
        }
        for (int i = 16; i < 80; ++i) w[i] = Rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            const uint32_t f = i < 20 ? (b & c) | (~b & d) : i < 40 ? b ^ c ^ d : i < 60 ? (b & c) | (b & d) | (c & d) : b ^ c ^ d;
            const uint32_t k = i < 20 ? 0x5A827999 : i < 40 ? 0x6ED9EBA1 : i < 60 ? 0x8F1BBCDC : 0xCA62C1D6;
            const uint32_t t = Rol(a, 5) + f + e + k + w[i];
            e = d, d = c, c = Rol(b, 30), b = a, a = t;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e;
    }
    void Add(const uint8_t* p, size_t n) {
        total += n;
        while (n) {
            const size_t take = std::min(n, 64 - used);
            std::memcpy(block + used, p, take);
            used += take, p += take, n -= take;
            if (used == 64) Compress(), used = 0;
        }
    }
    std::string Hex() { // (finishes)
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80, zero = 0;
        Add(&one, 1);
        while (used != 56) Add(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
        Add(len, 8);
        static const char* d = "0123456789abcdef";
        std::string s;
        for (uint32_t v : h) {
            for (int i = 28; i >= 0; i -= 4) s += d[(v >> i) & 15];
        }
        return s;
    }
};

// --- LZ4 (a shader entry's packing) --------------------------------------------------------------------------------

/// A plain LZ4 block decoded to exactly `size` bytes; false when it does not decode so (shaderseed.py lz4_block).
bool Lz4(const uint8_t* in, size_t in_size, size_t size, Bytes& out) {
    out.clear();
    out.reserve(size);
    const uint8_t* end = in + in_size;
    const auto length = [&](size_t n, bool& ok) {
        if (n == 15) {
            for (uint8_t b = 255; b == 255;) {
                if (in >= end) return ok = false, n;
                b = *in++;
                n += b;
            }
        }
        return n;
    };
    while (in < end) {
        bool ok = true;
        const uint8_t token = *in++;
        const size_t lit = length(token >> 4, ok);
        if (!ok || lit > static_cast<size_t>(end - in)) return false;
        out.insert(out.end(), in, in + lit);
        in += lit;
        if (in >= end) break;
        if (end - in < 2) return false;
        const size_t back = in[0] | in[1] << 8;
        in += 2;
        const size_t match = length(token & 15, ok) + 4;
        if (!ok || back == 0 || back > out.size() || out.size() + match > size) return false;
        for (size_t i = 0, from = out.size() - back; i < match; ++i) out.push_back(out[from + i]);
        if (out.size() > size) return false;
    }
    return out.size() == size;
}

// --- The entries and the stages -----------------------------------------------------------------------------------

constexpr uint32_t kControlMagic = 0x98761234;
constexpr uint32_t kTagMagics[2] = {0x12345678, 0x12345679};

int32_t I32(const uint8_t* p) {
    int32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
uint32_t U32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

size_t Find(const Bytes& b, const void* what, size_t n, size_t from = 0) {
    if (from > b.size()) return std::string::npos;
    const auto it = std::search(b.begin() + static_cast<std::ptrdiff_t>(from), b.end(), static_cast<const uint8_t*>(what),
                                static_cast<const uint8_t*>(what) + n);
    return it == b.end() ? std::string::npos : static_cast<size_t>(it - b.begin());
}

struct Shared {
    std::mutex mu;
    std::set<std::string> seen;    // packed entries (SHA-1)
    std::set<std::string> written; // stage names
    std::string out;
    ExtractStats stats;
    std::string problem;
};

/// (code memory, control section, assembly text) of an unpacked entry: shaderseed.py split.
void TakeBlob(Shared& sh, const Bytes& blob) {
    size_t tag = std::string::npos;
    for (uint32_t m : kTagMagics) tag = std::min(tag, Find(blob, &m, 4));
    const size_t ctl = Find(blob, &kControlMagic, 4);
    const auto unsplit = [&] {
        std::lock_guard lock(sh.mu);
        ++sh.stats.entries, ++sh.stats.unsplit;
    };
    if (tag == std::string::npos || ctl == std::string::npos || ctl + 0x800 > blob.size()) return unsplit();
    const uint8_t* c = blob.data() + ctl;
    const uint32_t code_size = U32(c + 0x704);
    if (U32(c + 0xC) != 0x120 || U32(c + 0x10) != 0xB || (U32(c + 0x708) != 0x30 && U32(c + 0x708) != 0x100) ||
        tag + code_size > ctl) {
        return unsplit();
    }
    std::string text;
    for (const char* magic : {"!!NVvp5.0", "!!NVfp5.0", "!!NVgp5.0", "!!NVcp5.0", "!!NVtcp5.0", "!!NVtep5.0"}) {
        const size_t i = Find(blob, magic, std::strlen(magic));
        if (i == std::string::npos) continue;
        size_t j = i;
        while (j < blob.size() && blob[j]) ++j;
        text.assign(blob.begin() + static_cast<std::ptrdiff_t>(i), blob.begin() + static_cast<std::ptrdiff_t>(j));
        break;
    }
    Sha1 h;
    h.Add(blob.data() + tag, code_size);
    h.Add(c, 0x800);
    const std::string name = h.Hex().substr(0, 16);
    {
        std::lock_guard lock(sh.mu);
        ++sh.stats.entries;
        if (!sh.written.insert(name).second) return;
        ++sh.stats.stages;
    }
    const fs::path base = fs::path(sh.out) / name;
    std::ofstream(fs::path(base).replace_extension(".code"), std::ios::binary)
        .write(reinterpret_cast<const char*>(blob.data() + tag), code_size);
    std::ofstream(fs::path(base).replace_extension(".ctrl"), std::ios::binary).write(reinterpret_cast<const char*>(c), 0x800);
    std::ofstream(fs::path(base).replace_extension(".asm"), std::ios::binary).write(text.data(), static_cast<std::streamsize>(text.size()));
}

/// The entries of one chunk (or file): shaderseed.py entries. An entry's packed bytes start with LZ4's first token, its
/// literals with "LSLGS" (a stored entry: "LSLGS" at once), so each "LSLGS" is tried as the start of the literals
/// after a token with 0-4 extra length bytes.
void Scan(Shared& sh, const Bytes& data) {
    static const char kLslgs[] = "LSLGS";
    Bytes blob;
    for (size_t at = Find(data, kLslgs, 5); at != std::string::npos; at = Find(data, kLslgs, 5, at + 5)) {
        for (int k = -1; k < 5; ++k) { // -1: stored, not packed
            const long long start = k < 0 ? static_cast<long long>(at) : static_cast<long long>(at) - 1 - k;
            if (start < 4) continue;
            const long long n = I32(&data[start - 4]);
            if (n < 16 || n > static_cast<long long>(data.size()) - start - 5) continue;
            const int32_t u = I32(&data[start + n]);
            const uint8_t freq = data[start + n + 4];
            if (freq >= 16 || !(k < 0 ? u == n : n < u && u < 64 * n)) continue;
            Sha1 hp;
            hp.Add(&data[start], static_cast<size_t>(n));
            const std::string ph = hp.Hex();
            {
                std::lock_guard lock(sh.mu);
                if (sh.seen.count(ph)) break;
            }
            bool ok;
            if (k < 0) {
                blob.assign(data.begin() + start, data.begin() + start + n);
                ok = true;
            } else {
                ok = Lz4(&data[start], static_cast<size_t>(n), static_cast<size_t>(u), blob);
            }
            if (ok && blob.size() >= 5 && std::memcmp(blob.data(), kLslgs, 5) == 0) {
                {
                    std::lock_guard lock(sh.mu);
                    if (!sh.seen.insert(ph).second) break;
                }
                TakeBlob(sh, blob);
                break;
            }
        }
    }
}

// --- IoStore containers (.utoc + .ucas, TOC version 3, Unreal 4.27) ---------------------------------------------

uint64_t Be40(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 5; ++i) v = v << 8 | p[i];
    return v;
}

struct Container {
    fs::path ucas;
    uint32_t block_size = 0;
    struct Block {
        uint64_t offset;
        uint32_t csize, usize;
        uint8_t method;
    };
    std::vector<Block> blocks;
    struct Chunk {
        uint64_t offset, length; // in the uncompressed space
        uint8_t sha1[20];        // of its unpacked bytes (the TOC's chunk metas)
    };
    std::vector<Chunk> chunks; // by offset
    std::vector<std::string> methods;                    // [index - 1]
};

/// The TOC: header (144 bytes), chunk ids (12 each), offsets + lengths (10, big-endian 40-bit), perfect-hash seeds,
/// compression blocks (12: 40-bit offset, 24-bit compressed and uncompressed sizes, method index), method names (32).
bool ReadToc(const fs::path& utoc, Container& c, std::string* why) {
    std::ifstream f(utoc, std::ios::binary);
    Bytes d((std::istreambuf_iterator<char>(f)), {});
    const auto bad = [&](const std::string& what) {
        *why = utoc.string() + ": " + what;
        return false;
    };
    if (d.size() < 144 || std::memcmp(d.data(), "-==--==--==--==-", 16) != 0) return bad("not an IoStore TOC");
    const uint8_t version = d[16];
    const uint32_t header = U32(&d[20]), entries = U32(&d[24]), nblocks = U32(&d[28]), block_entry = U32(&d[32]),
                   nmethods = U32(&d[36]), method_len = U32(&d[40]);
    c.block_size = U32(&d[44]);
    const uint8_t flags = d[80];
    const uint32_t seeds = version >= 3 ? U32(&d[84]) : 0;
    if (version < 2 || version > 3) return bad("TOC version " + std::to_string(version) + " (expected 2-3: Unreal 4.26-4.27)");
    if (flags & 2) return bad("encrypted");
    if (block_entry != 12) return bad("unexpected block entry size");
    size_t at = header + size_t{entries} * 12;
    const size_t offsets = at;
    at += size_t{entries} * 10 + size_t{seeds} * 4;
    const size_t blocks = at;
    at += size_t{nblocks} * 12;
    if (at + size_t{nmethods} * method_len > d.size()) return bad("truncated");
    for (uint32_t i = 0; i < nmethods; ++i) {
        const char* s = reinterpret_cast<const char*>(&d[at + size_t{i} * method_len]);
        c.methods.emplace_back(s, strnlen(s, method_len));
    }
    for (uint32_t i = 0; i < nblocks; ++i) {
        const uint8_t* e = &d[blocks + size_t{i} * 12];
        uint64_t off = 0;
        std::memcpy(&off, e, 5);
        c.blocks.push_back({off, U32(e + 4) >> 8 & 0xFFFFFF, U32(e + 8) & 0xFFFFFF, e[11]});
    }
    const size_t metas = d.size() - size_t{entries} * 33; // (the TOC ends with one 33-byte meta per chunk: hash + flags)
    for (uint32_t i = 0; i < entries; ++i) {
        const uint8_t* e = &d[offsets + size_t{i} * 10];
        if (const uint64_t len = Be40(e + 5)) {
            Container::Chunk ch{Be40(e), len, {}};
            std::memcpy(ch.sha1, &d[metas + size_t{i} * 33], 20);
            c.chunks.push_back(ch);
        }
    }
    std::sort(c.chunks.begin(), c.chunks.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
    c.ucas = fs::path(utoc).replace_extension(".ucas");
    return true;
}

/// One compression block, unpacked (Oodle, or stored).
bool ReadBlock(std::ifstream& f, const Container& c, size_t i, Bytes& raw, Bytes& out, std::string* why) {
    const Container::Block& b = c.blocks[i];
    raw.assign(size_t{b.csize} + kOodleSlack, 0); // (the decoder may read past its input too: zeros, not stale bytes)
    f.seekg(static_cast<std::streamoff>(b.offset));
    if (!f.read(reinterpret_cast<char*>(raw.data()), b.csize)) return *why = c.ucas.string() + ": read failed", false;
    if (b.method == 0) {
        out.assign(raw.begin(), raw.begin() + b.csize);
        return true;
    }
    const std::string& m = b.method <= c.methods.size() ? c.methods[b.method - 1] : std::string("?");
    if (m != "Oodle") return *why = c.ucas.string() + ": blocks packed with " + m + " (only Oodle is read)", false;
    out.assign(size_t{b.usize} + kOodleSlack, 0);
    if (!OodleDecompress(raw.data(), b.csize, out.data(), b.usize)) {
        return *why = c.ucas.string() + ": block " + std::to_string(i) + " did not decompress", false;
    }
    out.resize(b.usize);
    return true;
}

// --- The .pak's global shader cache (pak version 11) ---------------------------------------------------------------

struct Reader {
    const Bytes& b;
    size_t o = 0;
    bool ok = true;
    template <typename T> T Get() {
        T v{};
        if (o + sizeof(T) > b.size()) return ok = false, v;
        std::memcpy(&v, &b[o], sizeof(T));
        o += sizeof(T);
        return v;
    }
    std::string Str() { // FString: a negative length is UTF-16 (only ASCII kept)
        const int32_t n = Get<int32_t>();
        std::string s;
        if (n < 0) {
            for (int32_t i = 0; i < -n && o + 2 <= b.size(); ++i, o += 2) s += static_cast<char>(b[o]);
        } else {
            if (o + static_cast<size_t>(n) > b.size()) return ok = false, s;
            s.assign(reinterpret_cast<const char*>(&b[o]), static_cast<size_t>(n));
            o += static_cast<size_t>(n);
        }
        while (!s.empty() && s.back() == 0) s.pop_back();
        return s;
    }
    void Skip(size_t n) { o += n; }
};

/// The unpacked file of `pak` whose path ends in `suffix` ("" when there is none): shaderseed.py pak_file.
bool PakFile(const fs::path& pak, const std::string& suffix, Bytes& out, std::string* why) {
    std::ifstream f(pak, std::ios::binary);
    f.seekg(0, std::ios::end);
    const long long end = f.tellg();
    constexpr int kFooter = 16 + 1 + 4 + 4 + 8 + 8 + 20 + 5 * 32;
    if (end < kFooter) return true;
    Bytes foot(kFooter);
    f.seekg(end - kFooter);
    f.read(reinterpret_cast<char*>(foot.data()), kFooter);
    if (U32(&foot[17]) != 0x5A6F12E1 || U32(&foot[21]) != 11) return true; // (not a version 11 pak: nothing here)
    long long index_off, index_size;
    std::memcpy(&index_off, &foot[25], 8);
    std::memcpy(&index_size, &foot[33], 8);
    std::vector<std::string> methods;
    for (int i = 0; i < 5; ++i) {
        const char* s = reinterpret_cast<const char*>(&foot[61 + 32 * i]);
        methods.emplace_back(s, strnlen(s, 32));
    }
    const auto read = [&](long long off, long long size) {
        Bytes b(static_cast<size_t>(size));
        f.seekg(off);
        f.read(reinterpret_cast<char*>(b.data()), size);
        return b;
    };
    const Bytes ix = read(index_off, index_size);
    Reader r{ix};
    const std::string mount = r.Str();
    r.Get<int32_t>();
    r.Get<uint64_t>();
    if (r.Get<uint32_t>()) r.Skip(8 + 8 + 20);
    if (!r.Get<uint32_t>()) return *why = pak.string() + ": no full directory index", false;
    const long long full_off = r.Get<long long>(), full_size = r.Get<long long>();
    r.Skip(20);
    const int32_t encoded_size = r.Get<int32_t>();
    if (!r.ok || r.o + static_cast<size_t>(encoded_size) > ix.size()) return *why = pak.string() + ": bad index", false;
    const Bytes encoded(ix.begin() + static_cast<std::ptrdiff_t>(r.o), ix.begin() + static_cast<std::ptrdiff_t>(r.o) + encoded_size);
    const Bytes dir = read(full_off, full_size);
    Reader dr{dir};
    long long where = -1;
    for (int32_t i = dr.Get<int32_t>(); i > 0 && dr.ok; --i) {
        const std::string dname = dr.Str();
        for (int32_t j = dr.Get<int32_t>(); j > 0 && dr.ok; --j) {
            const std::string fname = dr.Str();
            const int32_t loc = dr.Get<int32_t>();
            const std::string path = mount + dname + fname;
            if (path.size() >= suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) where = loc;
        }
    }
    if (where < 0) return true;
    Reader e{encoded, static_cast<size_t>(where)};
    const uint32_t v = e.Get<uint32_t>();
    const uint32_t block_size = (v & 0x3F) == 0x3F ? e.Get<uint32_t>() : (v & 0x3F) << 11;
    const uint32_t method = (v >> 23) & 0x3F, nblocks = (v >> 6) & 0xFFFF;
    const long long off = v & (1u << 31) ? e.Get<uint32_t>() : e.Get<long long>();
    const long long usize = v & (1u << 30) ? e.Get<uint32_t>() : e.Get<long long>();
    const long long size = method ? (v & (1u << 29) ? e.Get<uint32_t>() : e.Get<long long>()) : usize;
    std::vector<uint32_t> sizes;
    if (method) {
        if (nblocks == 1) sizes.push_back(static_cast<uint32_t>(size));
        else for (uint32_t i = 0; i < nblocks; ++i) sizes.push_back(e.Get<uint32_t>());
    }
    if ((v >> 22) & 1) return *why = pak.string() + ": " + suffix + " is encrypted", false;
    const long long header = 8 + 8 + 8 + 4 + 20 + (method ? 4 + 16 * nblocks : 0) + 1 + 4;
    f.seekg(off + header);
    if (!method) {
        out = read(off + header, usize);
        return true;
    }
    const std::string name = method <= methods.size() ? methods[method - 1] : "?";
    if (name != "Oodle") return *why = pak.string() + ": " + suffix + " is packed with " + name, false;
    out.clear();
    Bytes raw, part;
    for (uint32_t s : sizes) {
        raw.assign(size_t{s} + kOodleSlack, 0);
        f.read(reinterpret_cast<char*>(raw.data()), s);
        const size_t want = std::min<size_t>(block_size, static_cast<size_t>(usize) - out.size());
        part.resize(want + kOodleSlack);
        if (!OodleDecompress(raw.data(), s, part.data(), want)) {
            return *why = pak.string() + ": a block of " + suffix + " did not decompress", false;
        }
        out.insert(out.end(), part.begin(), part.begin() + static_cast<std::ptrdiff_t>(want));
    }
    return true;
}

} // namespace

bool ExtractStages(const std::string& romfs, const std::string& out, int threads,
                   const std::function<void(double)>& progress, const std::atomic<bool>& cancel, ExtractStats* stats,
                   std::string* why) {
    std::error_code ec;
    fs::create_directories(fs::path(out), ec);
    Shared sh;
    sh.out = out;
    for (const auto& e : fs::directory_iterator(fs::path(out), ec)) { // (a resumed run: what is there stays)
        if (e.path().extension() == ".ctrl") sh.written.insert(e.path().stem().string());
    }
    std::vector<fs::path> tocs, paks;
    for (const auto& e : fs::recursive_directory_iterator(fs::path(romfs), ec)) {
        if (e.path().extension() == ".utoc") tocs.push_back(e.path());
        if (e.path().extension() == ".pak") paks.push_back(e.path());
    }
    std::sort(tocs.begin(), tocs.end());
    std::sort(paks.begin(), paks.end());
    if (tocs.empty() && paks.empty()) return *why = "no .utoc or .pak in " + romfs, false;

    std::vector<Container> containers(tocs.size());
    uint64_t total = 0;
    for (size_t i = 0; i < tocs.size(); ++i) {
        if (!ReadToc(tocs[i], containers[i], why)) return false;
        for (const auto& b : containers[i].blocks) total += b.csize;
        sh.stats.containers++;
    }
    std::atomic<uint64_t> done{0};
    for (Container& c : containers) {
        std::atomic<size_t> next{0};
        std::vector<std::thread> workers;
        for (int t = 0; t < std::max(1, threads); ++t) {
            workers.emplace_back([&] {
                std::ifstream f(c.ucas, std::ios::binary);
                Bytes raw, block, chunk;
                size_t have = SIZE_MAX; // the block in `block`
                std::string problem;
                for (size_t i; !cancel && (i = next++) < c.chunks.size();) {
                    const uint64_t off = c.chunks[i].offset, len = c.chunks[i].length;
                    chunk.clear();
                    for (uint64_t b = off / c.block_size; b <= (off + len - 1) / c.block_size; ++b) {
                        if (b >= c.blocks.size()) {
                            problem = c.ucas.string() + ": a chunk runs past the last block";
                            break;
                        }
                        if (b != have) {
                            if (!f.good()) f.clear();
                            if (!ReadBlock(f, c, b, raw, block, &problem)) break;
                            have = b;
                            done += c.blocks[b].csize;
                        }
                        const uint64_t from = b == off / c.block_size ? off % c.block_size : 0;
                        const uint64_t to = std::min<uint64_t>(block.size(), off + len - b * c.block_size);
                        if (from < to) chunk.insert(chunk.end(), block.begin() + from, block.begin() + to);
                    }
                    if (!problem.empty()) {
                        std::lock_guard lock(sh.mu);
                        if (sh.problem.empty()) sh.problem = problem;
                        break;
                    }
                    { // the chunk against the TOC's SHA-1: a wrongly unpacked chunk is counted and still scanned (its
                      // stages mostly come out right; a garbled one fails in the seed's translation, which skips it).
                      // 2026-11-01: GCC -O2 miscompiled ooz - 1,177 chunks wrong; built as CMake has it now, 0.
                        Sha1 h;
                        h.Add(chunk.data(), chunk.size());
                        static const char* hex = "0123456789abcdef";
                        std::string want;
                        for (uint8_t x : c.chunks[i].sha1) want += hex[x >> 4], want += hex[x & 15];
                        if (h.Hex() != want) {
                            std::lock_guard lock(sh.mu);
                            ++sh.stats.bad_chunks;
                        }
                    }
                    Scan(sh, chunk);
                    {
                        std::lock_guard lock(sh.mu);
                        ++sh.stats.chunks;
                    }
                    if (progress && total) progress(static_cast<double>(done) / static_cast<double>(total));
                }
            });
        }
        for (auto& w : workers) w.join();
        if (!sh.problem.empty()) return *why = sh.problem, false;
        if (cancel) return *why = "cancelled", false;
    }
    for (const fs::path& pak : paks) { // the global shaders
        Bytes cache;
        if (!PakFile(pak, "GlobalShaderCache-SF_GLSL_SWITCH.bin", cache, why)) return false;
        if (!cache.empty()) Scan(sh, cache);
    }
    if (progress) progress(1.0);
    if (stats) *stats = sh.stats;
    return true;
}

} // namespace rt::seed
