// The player's dump unpacked (unpack.h).
#include "unpack.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "../platform/platform.h"

namespace fs = std::filesystem;

namespace rt::build {
namespace {

std::string Utf8(const fs::path& p) {
    const std::u8string s = p.u8string();
    return std::string(s.begin(), s.end());
}

/// A PFS0 container (an .nsp): its files, their offsets counted from the end of the header (16 bytes + 24 per entry
/// + the string table - not from the file's start: README "1.0.1").
struct Nsp {
    fs::path file;
    struct Entry {
        std::string name;
        uint64_t offset, size; // absolute in the file
    };
    std::vector<Entry> entries;
    const Entry* program = nullptr; // the largest .nca: the program's (the others: control, legal, meta)
    const Entry* ticket = nullptr;  // the .tik (rights-ID crypto), if any
};

bool ReadNsp(const fs::path& file, Nsp& n, std::string* why) {
    n.file = file;
    std::ifstream f(file, std::ios::binary);
    uint8_t h[16];
    if (!f.read(reinterpret_cast<char*>(h), 16) || std::memcmp(h, "PFS0", 4) != 0) {
        return *why = file.filename().string() + " is not an .nsp (PFS0) file", false;
    }
    uint32_t count, strings;
    std::memcpy(&count, h + 4, 4);
    std::memcpy(&strings, h + 8, 4);
    if (count > 4096 || strings > (1u << 20)) return *why = file.filename().string() + ": damaged header", false;
    std::vector<uint8_t> e(size_t{count} * 24), s(strings);
    f.read(reinterpret_cast<char*>(e.data()), static_cast<std::streamsize>(e.size()));
    f.read(reinterpret_cast<char*>(s.data()), static_cast<std::streamsize>(s.size()));
    if (!f) return *why = file.filename().string() + ": damaged header", false;
    const uint64_t data = 16 + uint64_t{count} * 24 + strings;
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t off, size;
        uint32_t name;
        std::memcpy(&off, &e[i * 24], 8);
        std::memcpy(&size, &e[i * 24 + 8], 8);
        std::memcpy(&name, &e[i * 24 + 16], 4);
        if (name >= strings) return *why = file.filename().string() + ": damaged header", false;
        n.entries.push_back({std::string(reinterpret_cast<const char*>(&s[name]), strnlen(reinterpret_cast<const char*>(&s[name]), strings - name)),
                             data + off, size});
    }
    for (const Nsp::Entry& x : n.entries) {
        const std::string ext = fs::path(x.name).extension().string();
        const bool meta = x.name.size() > 9 && x.name.compare(x.name.size() - 9, 9, ".cnmt.nca") == 0;
        if (ext == ".nca" && !meta && (!n.program || x.size > n.program->size)) n.program = &x;
        if (ext == ".tik") n.ticket = &x;
    }
    if (!n.program) return *why = file.filename().string() + " holds no program", false;
    return true;
}

std::vector<fs::path> NspFiles(const std::string& source) {
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(fs::path(source), ec)) {
        std::string ext = e.path().extension().string();
        for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (e.is_regular_file(ec) && ext == ".nsp") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

/// What the unpacked folder was made from: each .nsp's name and size (<game>/unpacked.txt).
std::string Stamp(const std::string& source) {
    std::string s;
    std::error_code ec;
    for (const fs::path& f : NspFiles(source)) s += f.filename().string() + " " + std::to_string(fs::file_size(f, ec)) + "\n";
    return s;
}

std::string ReadText(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), {});
}

/// hactool's output with every key value masked (it echoes values in its warnings and errors): no run of 32 or more hex
/// digits survives; the key names and the rest of the line stay (a failure's reason, the player's report).
std::string WithoutKeys(const std::string& text) {
    std::string out;
    for (size_t i = 0; i < text.size();) {
        size_t run = 0;
        while (i + run < text.size() && std::isxdigit(static_cast<unsigned char>(text[i + run]))) ++run;
        if (run >= 32) out += "<key>";
        else out.append(text, i, std::max<size_t>(run, 1));
        i += std::max<size_t>(run, 1);
    }
    return out;
}

/// prod.keys as hactool reads it. Some dumpers (seen 2026-11-01 with a newer console's dump) write the *_kek_source
/// keys with one byte too many - 34 hex digits, the same last byte on all of them, the first 32 the key (checked
/// against an older dump of the same console: 27 of 27 equal) - and hactool stops at the first ("must be 32 hex
/// digits"); no key hactool reads has 34 digits. Those are cut to 32 in a copy in `work` (removed with it); `fixed`
/// gets their names. The player's file itself when nothing needed it.
fs::path KeysForHactool(const fs::path& keys, const fs::path& work, std::vector<std::string>& fixed) {
    std::ifstream in(keys, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), {}), out, line;
    const auto trim = [](std::string v) {
        const size_t b = v.find_first_not_of(" \t\r"), e = v.find_last_not_of(" \t\r");
        return b == std::string::npos ? std::string() : v.substr(b, e - b + 1);
    };
    for (size_t i = 0; i <= text.size(); ++i) {
        if (i < text.size() && text[i] != '\n') {
            line += text[i];
            continue;
        }
        const size_t eq = line.find('=');
        const std::string value = eq == std::string::npos ? std::string() : trim(line.substr(eq + 1));
        const bool hex = !value.empty() && std::all_of(value.begin(), value.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; });
        if (hex && value.size() == 34) {
            const std::string name = trim(line.substr(0, eq));
            fixed.push_back(name);
            line = name + " = " + value.substr(0, 32);
        }
        out += line + (i < text.size() ? "\n" : "");
        line.clear();
    }
    if (fixed.empty()) return keys;
    const fs::path copy = work / "prod.keys";
    std::ofstream(copy, std::ios::binary | std::ios::trunc) << out;
    return copy;
}

uint64_t TreeSize(const fs::path& dir) {
    uint64_t total = 0;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec)) {
        if (e.is_regular_file(ec)) total += e.file_size(ec);
    }
    return total;
}

} // namespace

bool IsPacked(const std::string& source) {
    return !source.empty() && !NspFiles(source).empty();
}

bool UnpackNeeded(const std::string& source, const std::string& game) {
    std::error_code ec;
    return !fs::exists(fs::path(game) / "exefs" / "main", ec) || ReadText(fs::path(game) / "unpacked.txt") != Stamp(source);
}

bool Unpack(const UnpackArgs& a, const std::function<void(double, const char*)>& progress,
            const std::atomic<bool>& cancel, std::string* why) {
    std::error_code ec;
    // the files: the base game (the largest program) and its update (the next, with a ticket), as dumped
    std::vector<Nsp> nsps;
    for (const fs::path& f : NspFiles(a.source)) {
        Nsp n;
        if (!ReadNsp(f, n, why)) return false;
        nsps.push_back(std::move(n));
    }
    if (nsps.empty()) return *why = "No .nsp files in " + a.source + ".", false;
    std::sort(nsps.begin(), nsps.end(), [](const Nsp& x, const Nsp& y) { return x.program->size > y.program->size; });
    const Nsp& base = nsps[0];
    const Nsp* update = nsps.size() > 1 ? &nsps[1] : nullptr;
    fs::path keys = a.keys;
    if (fs::is_directory(keys, ec)) keys /= "prod.keys";
    if (a.keys.empty() || !fs::exists(keys, ec)) return *why = "Choose your keys first (Settings: Keys - the folder with prod.keys).", false;

    // room: the cut-out NCAs, then the unpacked game (about the base program's size) and some to spare
    const uint64_t temp = base.program->size + (update ? update->program->size : 0);
    const uint64_t need = temp + base.program->size + (2ull << 30);
    fs::create_directories(fs::path(a.game).parent_path(), ec);
    const fs::space_info room = fs::space(fs::path(a.game).parent_path(), ec);
    if (!ec && room.available < need) {
        char b[160];
        std::snprintf(b, sizeof b, "Not enough free disk space to unpack: %.1f GB needed, %.1f GB free.", need / 1e9,
                      room.available / 1e9);
        return *why = b, false;
    }
    fs::remove_all(fs::path(a.game), ec);
    fs::create_directories(fs::path(a.game), ec); // (hactool makes only the last folder of --exefsdir / --romfsdir)
    fs::create_directories(fs::path(a.work), ec);

    // 1. the program NCAs cut out (hactool reads files, not entries of an .nsp)
    const auto cut = [&](const Nsp& n, const fs::path& to, uint64_t& copied) {
        std::ifstream in(n.file, std::ios::binary);
        std::ofstream out(to, std::ios::binary | std::ios::trunc);
        in.seekg(static_cast<std::streamoff>(n.program->offset));
        std::vector<char> buf(8 << 20);
        for (uint64_t left = n.program->size; left;) {
            if (cancel) return *why = "cancelled", false;
            const size_t k = static_cast<size_t>(std::min<uint64_t>(left, buf.size()));
            if (!in.read(buf.data(), static_cast<std::streamsize>(k)) || !out.write(buf.data(), static_cast<std::streamsize>(k))) {
                return *why = "Could not copy out of " + n.file.filename().string() + " (disk full?)", false;
            }
            left -= k, copied += k;
            progress(0.4 * static_cast<double>(copied) / static_cast<double>(temp), "Reading the game files");
        }
        return true;
    };
    const fs::path base_nca = fs::path(a.work) / "base.nca", update_nca = fs::path(a.work) / "update.nca";
    uint64_t copied = 0;
    const auto clean = [&] { fs::remove_all(fs::path(a.work), ec); };
    if (!cut(base, base_nca, copied) || (update && !cut(*update, update_nca, copied))) return clean(), false;

    // 2. hactool: the update over the base (or the base alone)
    std::vector<std::string> fixed;
    const fs::path hactool_keys = KeysForHactool(keys, fs::path(a.work), fixed);
    std::string note; // (the log's first line: the keys cut to 32 digits, by name)
    for (const std::string& n : fixed) note += (note.empty() ? "keys: 34 hex digits cut to 32 (a dumper's extra byte): " : ", ") + n;
    if (!note.empty()) note += "\n";
    std::vector<std::string> args{a.hactool, "-k", Utf8(hactool_keys), "--exefsdir=" + Utf8(fs::path(a.game) / "exefs"),
                                  "--romfsdir=" + Utf8(fs::path(a.game) / "romfs")};
    const Nsp& top = update ? *update : base;
    if (top.ticket) {
        std::ifstream t(top.file, std::ios::binary);
        t.seekg(static_cast<std::streamoff>(top.ticket->offset + 0x180));
        uint8_t key[16];
        if (top.ticket->size < 0x190 || !t.read(reinterpret_cast<char*>(key), 16)) return clean(), *why = "Damaged ticket in " + top.file.filename().string(), false;
        static const char* hex = "0123456789abcdef";
        std::string k;
        for (const uint8_t x : key) k += hex[x >> 4], k += hex[x & 15];
        args.push_back("--titlekey=" + k);
    }
    if (update) args.push_back("--basenca=" + Utf8(base_nca));
    args.push_back(Utf8(update ? update_nca : base_nca));
    const fs::path raw = fs::path(a.work) / "hactool.out";
    std::string error;
    platform::Child* c = platform::StartChild(args, Utf8(raw), &error);
    if (!c) return clean(), *why = "Could not start hactool (" + error + "). If antivirus blocked it, allow the tools folder.", false;
    int code = 0;
    for (int tick = 0; !platform::ChildDone(c, &code); ++tick) {
        if (cancel) {
            platform::KillChild(c);
            platform::FreeChild(c);
            clean();
            return *why = "cancelled", false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (tick % 20 == 0) {
            const double done = static_cast<double>(TreeSize(fs::path(a.game))) / static_cast<double>(base.program->size);
            progress(0.4 + 0.6 * std::min(done, 0.99), "Unpacking the game files");
        }
    }
    platform::FreeChild(c);
    const std::string out = note + WithoutKeys(ReadText(raw));
    std::ofstream(fs::path(a.log), std::ios::binary | std::ios::trunc) << out;
    clean(); // (hactool's own output with the keys goes with it)
    if (code != 0 || !fs::exists(fs::path(a.game) / "exefs" / "main", ec) || TreeSize(fs::path(a.game) / "romfs") == 0) {
        std::string last; // (its last non-empty line)
        const size_t end = out.find_last_not_of("\r\n");
        if (end != std::string::npos) {
            const size_t begin = out.rfind('\n', end);
            last = out.substr(begin == std::string::npos ? 0 : begin + 1, end - (begin == std::string::npos ? 0 : begin + 1) + 1);
        }
        *why = "Unpacking failed" + (last.empty() ? std::string(".") : ": " + last) +
               " Check that the keys match the game (prod.keys from the same console's firmware or newer). Log: " + a.log;
        fs::remove_all(fs::path(a.game), ec);
        return false;
    }
    std::ofstream(fs::path(a.game) / "unpacked.txt", std::ios::binary | std::ios::trunc) << Stamp(a.source);
    progress(1.0, "Unpacked");
    return true;
}

} // namespace rt::build
