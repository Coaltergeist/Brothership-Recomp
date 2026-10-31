#include "patch.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace nxrecomp {
namespace {

constexpr u64 kNsoHeader = 0x100; // IPS offsets of an NSO count its header (Atmosphère's loader)

std::runtime_error Error(const std::string& path, size_t line, const std::string& what) {
    return std::runtime_error(path + (line ? ":" + std::to_string(line) : "") + ": " + what);
}

std::string Lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

int HexDigit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
}

// Hex digits to bytes; an odd count is an error unless pad_odd (a build id: "…0" padded to a whole byte).
bool HexBytes(const std::string& hex, std::vector<u8>& out, bool pad_odd = false) {
    std::string h = hex;
    if (h.size() % 2) {
        if (!pad_odd) return false;
        h += '0';
    }
    out.clear();
    for (size_t i = 0; i < h.size(); i += 2) {
        const int hi = HexDigit(h[i]), lo = HexDigit(h[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back(static_cast<u8>(hi << 4 | lo));
    }
    return true;
}

std::string Hex(const std::vector<u8>& b, size_t max = SIZE_MAX) {
    std::string s;
    char buf[3];
    for (size_t i = 0; i < b.size() && i < max; ++i) {
        std::snprintf(buf, sizeof buf, "%02x", b[i]);
        s += buf;
    }
    return s;
}

std::string Trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

// IPSwitch text patch: `@nsobid-<hex>`, `@flag offset_shift <n>` (the file's offsets plus this are IPS
// offsets), `@enabled` / `@disabled` start a section, `@stop` ends the file; in an enabled section each line is
// `<hex offset> <hex bytes>` or `<hex offset> "string"`. `//` and `#` start comments.
PatchFile ParsePchtxt(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw Error(path, 0, "cannot read");
    PatchFile pf;
    pf.path = path;
    u64 shift = 0;
    enum { kNone, kEnabled, kDisabled } section = kNone;
    std::string raw;
    for (size_t n = 1; std::getline(f, raw); ++n) {
        // Cut a `//` comment outside quotes.
        bool quoted = false;
        for (size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] == '\\' && quoted) {
                ++i;
            } else if (raw[i] == '"') {
                quoted = !quoted;
            } else if (!quoted && raw.compare(i, 2, "//") == 0) {
                raw.resize(i);
                break;
            }
        }
        const std::string line = Trim(raw);
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == '@') {
            const std::string d = Lower(line);
            if (d.rfind("@nsobid-", 0) == 0) {
                if (!HexBytes(Trim(line.substr(8)), pf.build_id, true) || pf.build_id.empty() || pf.build_id.size() > 32) {
                    throw Error(path, n, "bad build id: " + line);
                }
            } else if (d.rfind("@flag", 0) == 0) {
                const std::string rest = Trim(line.substr(5));
                if (Lower(rest).rfind("offset_shift", 0) == 0) {
                    shift = std::stoull(Trim(rest.substr(12)), nullptr, 0);
                } // print_values, debug_info_offset, …: for IPSwitch's own output
            } else if (d == "@enabled") {
                section = kEnabled;
            } else if (d == "@disabled") {
                section = kDisabled;
            } else if (d == "@stop") {
                break;
            } else if (d == "@little-endian" || d == "@big-endian") {
                throw Error(path, n, line + ": not supported (values are written as bytes in file order)");
            } // anything else (an author's handle, …) is text
            continue;
        }
        if (section != kEnabled) continue; // free text, or a disabled section
        const size_t sp = line.find_first_of(" \t");
        if (sp == std::string::npos) throw Error(path, n, "expected `<offset> <value>`: " + line);
        PatchWrite w;
        size_t used = 0;
        const std::string off = line.substr(0, sp);
        try {
            w.ips_offset = std::stoull(off, &used, 16) + shift;
        } catch (const std::exception&) {
            used = 0;
        }
        if (used != off.size()) throw Error(path, n, "bad offset: " + off);
        const std::string value = Trim(line.substr(sp));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            for (size_t i = 1; i + 1 < value.size(); ++i) {
                char c = value[i];
                if (c == '\\' && i + 2 < value.size()) {
                    c = value[++i];
                    c = c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c == '0' ? '\0' : c;
                }
                w.bytes.push_back(static_cast<u8>(c));
            }
        } else {
            std::string hex;
            for (char c : value) {
                if (c != ' ' && c != '\t') hex += c;
            }
            if (!HexBytes(hex, w.bytes)) throw Error(path, n, "bad value (hex bytes or \"string\"): " + value);
        }
        if (w.bytes.empty()) throw Error(path, n, "empty value");
        pf.writes.push_back(std::move(w));
    }
    if (pf.build_id.empty()) throw Error(path, 0, "no @nsobid-<build id> line");
    return pf;
}

// IPS ("PATCH", 3-byte offsets, "EOF") and IPS32 ("IPS32", 4-byte offsets, "EEOF"): records of a big-endian
// offset and size, the bytes, or for size 0 a run (count, byte). The build id is the file name.
PatchFile ParseIps(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error(path, 0, "cannot read");
    const std::vector<u8> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    PatchFile pf;
    pf.path = path;
    const std::string stem = std::filesystem::path(path).stem().string();
    if (!HexBytes(stem, pf.build_id, true) || pf.build_id.empty() || pf.build_id.size() > 32) {
        throw Error(path, 0, "an .ips file is named after the build id it patches (<hex>.ips)");
    }
    const auto starts = [&](size_t at, const char* s) {
        const size_t n = std::char_traits<char>::length(s);
        return at + n <= d.size() && std::equal(s, s + n, d.begin() + at);
    };
    const bool ips32 = starts(0, "IPS32");
    if (!ips32 && !starts(0, "PATCH")) throw Error(path, 0, "not an IPS / IPS32 file");
    const size_t width = ips32 ? 4 : 3;
    const char* const end = ips32 ? "EEOF" : "EOF";
    const auto be = [&](size_t at, size_t n) {
        if (at + n > d.size()) throw Error(path, 0, "truncated");
        u64 v = 0;
        for (size_t i = 0; i < n; ++i) v = v << 8 | d[at + i];
        return v;
    };
    size_t at = 5;
    // The terminator, at the end (IPS may add a 3-byte truncation size after it).
    while (!(starts(at, end) && d.size() - at <= std::char_traits<char>::length(end) + 3)) {
        PatchWrite w;
        w.ips_offset = be(at, width);
        const u64 size = be(at + width, 2);
        at += width + 2;
        if (size) {
            if (at + size > d.size()) throw Error(path, 0, "truncated");
            w.bytes.assign(d.begin() + at, d.begin() + at + size);
            at += size;
        } else {
            w.bytes.assign(be(at, 2), static_cast<u8>(be(at + 2, 1)));
            at += 3;
        }
        if (!w.bytes.empty()) pf.writes.push_back(std::move(w));
    }
    return pf;
}

bool IsPatchFile(const std::filesystem::path& p) {
    const std::string ext = Lower(p.extension().string());
    return ext == ".pchtxt" || ext == ".ips";
}

PatchFile ParseFile(const std::filesystem::path& p) {
    return Lower(p.extension().string()) == ".pchtxt" ? ParsePchtxt(p.string()) : ParseIps(p.string());
}

} // namespace

std::vector<PatchFile> LoadPatches(const std::string& path) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    if (fs::is_directory(path)) {
        for (const auto& e : fs::recursive_directory_iterator(path)) {
            if (e.is_regular_file() && IsPatchFile(e.path())) files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) throw std::runtime_error(path + ": no .pchtxt or .ips files");
    } else if (fs::is_regular_file(path) && IsPatchFile(path)) {
        files.push_back(path);
    } else {
        throw std::runtime_error(path + ": not a .pchtxt / .ips file or a directory");
    }
    std::vector<PatchFile> out;
    for (const auto& f : files) {
        out.push_back(ParseFile(f));
        out.back().source = path;
    }
    return out;
}

std::string ApplyPatches(const std::vector<PatchFile>& patches, std::vector<Image>& modules) {
    std::string report, detail;
    std::vector<std::string> applied_sources;
    char buf[256];
    for (const PatchFile& pf : patches) {
        Image* m = nullptr;
        for (Image& img : modules) {
            if (!img.build_id.empty() && pf.build_id.size() <= img.build_id.size() &&
                std::equal(pf.build_id.begin(), pf.build_id.end(), img.build_id.begin())) {
                if (m) throw std::runtime_error(pf.path + ": build id matches both " + m->name + " and " + img.name);
                m = &img;
            }
        }
        if (!m) {
            std::snprintf(buf, sizeof buf, "patch %s: build id %s matches no module, skipped\n", pf.path.c_str(),
                          Hex(pf.build_id).c_str());
            report += buf;
            continue;
        }
        if (m->format != "nso" && m->format != "nro") {
            throw std::runtime_error(pf.path + ": " + m->name + " is an " + m->format + "; patches need an NSO or NRO");
        }
        const u64 header = m->format == "nso" ? kNsoHeader : 0;
        size_t bytes = 0, changed = 0, code = 0;
        for (const PatchWrite& w : pf.writes) {
            const u64 n = w.bytes.size();
            const Segment* seg = nullptr;
            if (w.ips_offset >= header) {
                const u64 off = w.ips_offset - header;
                for (const Segment& s : m->segments) {
                    if (off >= s.offset && off + n <= s.offset + s.file_size) seg = &s;
                }
            }
            if (!seg) {
                std::snprintf(buf, sizeof buf, ": a write at IPS offset 0x%llx (%llu bytes) is outside %s's segments",
                              (unsigned long long)w.ips_offset, (unsigned long long)n, m->name.c_str());
                throw std::runtime_error(pf.path + buf);
            }
            const u64 off = w.ips_offset - header;
            const std::vector<u8> old(m->mem.begin() + off, m->mem.begin() + off + n);
            std::copy(w.bytes.begin(), w.bytes.end(), m->mem.begin() + off);
            bytes += n;
            changed += !std::equal(old.begin(), old.end(), w.bytes.begin());
            code += seg->exec;
            std::snprintf(buf, sizeof buf, "  %s+0x%llx %s: %s -> %s\n", m->name.c_str(), (unsigned long long)off,
                          seg->exec ? "code" : seg->write ? "data" : "rodata", Hex(old, 32).c_str(),
                          Hex(w.bytes, 32).c_str());
            detail += buf;
        }
        std::snprintf(buf, sizeof buf, "patch %s: %s (build id %s…): %zu writes (%zu change something, %zu in code), %zu bytes\n",
                      pf.path.c_str(), m->name.c_str(), Hex(m->build_id, 8).c_str(), pf.writes.size(), changed, code,
                      bytes);
        report += buf;
        if (std::find(applied_sources.begin(), applied_sources.end(), pf.source) == applied_sources.end()) {
            applied_sources.push_back(pf.source);
        }
    }
    for (const PatchFile& pf : patches) {
        if (std::find(applied_sources.begin(), applied_sources.end(), pf.source) == applied_sources.end()) {
            throw std::runtime_error(report + pf.source + ": no patch matches a module's build id");
        }
    }
    return report + detail;
}

} // namespace nxrecomp
