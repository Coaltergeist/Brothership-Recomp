// fsp-srv (file system) and a virtual RomFS.
//
// The SDK reads its RomFS through an IStorage and parses the image itself, so
// the runtime serves a RomFS *image*. It is built from a directory of extracted
// files (RT_ROMFS): the header and the directory/file tables are generated in
// memory, and file data is read from the host files when the game asks for it,
// so nothing is copied. Layout: header, file data from 0x200 (each file
// 16-byte aligned), then the four tables. The format follows the documented
// RomFS (switchbrew "RomFS"; hactool, yuzu).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include <string>
#include <map>
#include <mutex>

#include "../ipc.h"
#include "../mods.h"
#include "../platform/platform.h"
#include "../npdm.h"
#include "../services.h"
#include "../threads.h"

namespace rt {
namespace {

// RT_TRACE_ROMFS=1 logs which RomFS file each read lands in.
const bool g_trace_romfs = std::getenv("RT_TRACE_ROMFS") != nullptr;
// RT_IO_STATS=1: a "romfs:" line every 100 IStorage reads (MB, time in all, the slowest).
// IStorage reads run without the kernel lock (2026-10-21): a 256 KB read took 1-1.6 ms through
// \\wsl.localhost and loading held the lock 500-630 ms a second with them, so the loader, task threads and
// GameThread waited 200-700 ms a second for it (README). The RomFS has its own lock; a file's first read
// is reported to hid (InputEvent) after the kernel lock is back. RT_ROMFS_LOCKED=1: read with it (A/B).
const bool g_romfs_locked = [] {
    const char* e = std::getenv("RT_ROMFS_LOCKED");
    return e && *e && *e != '0';
}();
const bool g_io_stats = [] {
    const char* e = std::getenv("RT_IO_STATS");
    return e && *e && *e != '0';
}();

namespace fs = std::filesystem;

// Save-data files (HostFile, HostFileSystem) are read and written without the kernel lock too (2026-10-24):
// in the new-game run every autosave froze the game 0.2-0.9 s, the saving PoolThread holding the lock in
// SendSyncRequest for up to 319 ms while the render, game and audio threads waited for it (the save folder is
// on \\wsl.localhost when run from WSL). SaveIo runs the host part of a request so (the request's guest
// buffers are read and written in it, as IStorage reads do) and logs one taking over 50 ms. RT_ROMFS_LOCKED=1
// keeps the lock for these too (A/B).
template <typename F>
void SaveIo(const char* what, const std::string& name, F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    if (g_romfs_locked) f();
    else RunWithoutGil(f);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ms > 50) Log("fs: slow save-data %s (%s): %.1f ms, without the kernel lock", what, name.c_str(), ms);
}

constexpr u32 kEmpty = 0xFFFF'FFFF;
constexpr u64 kDataOffset = 0x200;

u64 Align(u64 v, u64 a) { return (v + a - 1) & ~(a - 1); }

u32 PathHash(u32 parent, const std::string& name) {
    u32 hash = parent ^ 123456789;
    for (unsigned char c : name) {
        hash = (hash >> 5) | (hash << 27);
        hash ^= c;
    }
    return hash;
}

u32 HashTableCount(u32 entries) {
    if (entries < 3) {
        return 3;
    }
    if (entries < 19) {
        return entries | 1;
    }
    u32 n = entries;
    while (n % 2 == 0 || n % 3 == 0 || n % 5 == 0 || n % 7 == 0 || n % 11 == 0 || n % 13 == 0 || n % 17 == 0) {
        ++n;
    }
    return n;
}

class RomFs {
public:
    /// roots[0]: the game's extracted RomFS; the others: mods' romfs/ folders layered over it in order (B3): their
    /// directories merge, a file replaces the one at the same path (a later layer wins) or is added.
    explicit RomFs(const std::vector<std::string>& roots) {
        struct Dir {
            std::string name;
            u32 parent = 0, offset = 0;
            std::vector<u32> dirs, files;
            std::vector<std::pair<size_t, fs::path>> paths; // (layer, host directory) of every layer that has it
        };
        struct File {
            std::string name;
            u32 parent = 0, offset = 0;
            u64 data = 0, size = 0;
            fs::path path;
            size_t layer = 0;
        };
        std::vector<Dir> dirs(1);
        std::vector<File> files;
        dirs[0].paths.push_back({0, roots[0]});
        for (size_t l = 1; l < roots.size(); ++l) dirs[0].paths.push_back({l, roots[l]});
        std::vector<size_t> replaced(roots.size()), added(roots.size());
        // Breadth-first, names sorted, so the image is deterministic.
        for (size_t d = 0; d < dirs.size(); ++d) {
            struct Entry {
                bool dir = false;
                std::vector<std::pair<size_t, fs::path>> paths; // a directory's, per layer
                fs::path file;
                u64 size = 0;
                size_t layer = 0;
            };
            std::map<fs::path, Entry> entries; // by name, the layers merged
            for (const auto& [layer, path] : dirs[d].paths) {
                for (const auto& e : fs::directory_iterator(path)) {
                    const fs::path name = e.path().filename();
                    const bool dir = e.is_directory();
                    if (!dir && !e.is_regular_file()) continue;
                    auto [it, fresh] = entries.try_emplace(name);
                    Entry& x = it->second;
                    if (!fresh && x.dir != dir) {
                        Fatal("romfs: %s is a %s in one layer and a %s in %s", e.path().string().c_str(),
                              dir ? "directory" : "file", x.dir ? "directory" : "file", roots[x.dir ? x.paths.back().first : x.layer].c_str());
                    }
                    x.dir = dir;
                    if (dir) {
                        x.paths.push_back({layer, e.path()});
                    } else {
                        if (layer) ++(fresh ? added : replaced)[layer];
                        x.file = e.path();
                        x.size = e.file_size();
                        x.layer = layer;
                    }
                }
            }
            for (auto& [name, x] : entries) {
                if (x.dir) {
                    dirs[d].dirs.push_back(static_cast<u32>(dirs.size()));
                    dirs.push_back({name.string(), static_cast<u32>(d), 0, {}, {}, std::move(x.paths)});
                } else {
                    dirs[d].files.push_back(static_cast<u32>(files.size()));
                    files.push_back({name.string(), static_cast<u32>(d), 0, 0, x.size, x.file, x.layer});
                }
            }
        }
        // Table offsets and file data placement.
        u32 at = 0;
        for (auto& d : dirs) {
            d.offset = at;
            at += 0x18 + static_cast<u32>(Align(d.name.size(), 4));
        }
        const u32 dir_meta_size = at;
        at = 0;
        u64 data = 0;
        for (auto& f : files) {
            f.offset = at;
            at += 0x20 + static_cast<u32>(Align(f.name.size(), 4));
            data = Align(data, 0x10);
            f.data = data;
            data += f.size;
            const std::string host = f.path.string();
            files_.push_back({host, host.substr(roots[f.layer].size()), kDataOffset + f.data, f.size, {}});
        }
        const u32 file_meta_size = at;

        auto put32 = [](std::vector<u8>& v, u64 off, u32 x) { std::memcpy(v.data() + off, &x, 4); };
        auto put64 = [](std::vector<u8>& v, u64 off, u64 x) { std::memcpy(v.data() + off, &x, 8); };
        const u32 dir_buckets = HashTableCount(static_cast<u32>(dirs.size()));
        const u32 file_buckets = HashTableCount(static_cast<u32>(files.size()));
        std::vector<u8> dir_hash(dir_buckets * 4ULL, 0xFF), dir_meta(dir_meta_size);
        std::vector<u8> file_hash(file_buckets * 4ULL, 0xFF), file_meta(file_meta_size);
        auto bucket = [](std::vector<u8>& table, u32 index, u32 entry) {
            u32 prev;
            std::memcpy(&prev, table.data() + index * 4ULL, 4);
            std::memcpy(table.data() + index * 4ULL, &entry, 4);
            return prev;
        };
        for (const auto& d : dirs) {
            const u32 parent = dirs[d.parent].offset;
            u32 sibling = kEmpty;
            const auto& siblings = dirs[d.parent].dirs;
            if (&d != &dirs[0]) {
                auto it = std::find(siblings.begin(), siblings.end(), static_cast<u32>(&d - dirs.data()));
                if (it + 1 != siblings.end()) {
                    sibling = dirs[*(it + 1)].offset;
                }
            }
            put32(dir_meta, d.offset, parent);
            put32(dir_meta, d.offset + 4, sibling);
            put32(dir_meta, d.offset + 8, d.dirs.empty() ? kEmpty : dirs[d.dirs.front()].offset);
            put32(dir_meta, d.offset + 12, d.files.empty() ? kEmpty : files[d.files.front()].offset);
            put32(dir_meta, d.offset + 16, bucket(dir_hash, PathHash(parent, d.name) % dir_buckets, d.offset));
            put32(dir_meta, d.offset + 20, static_cast<u32>(d.name.size()));
            std::memcpy(dir_meta.data() + d.offset + 0x18, d.name.data(), d.name.size());
        }
        for (size_t i = 0; i < files.size(); ++i) {
            const auto& f = files[i];
            const auto& siblings = dirs[f.parent].files;
            auto it = std::find(siblings.begin(), siblings.end(), static_cast<u32>(i));
            const u32 parent = dirs[f.parent].offset;
            put32(file_meta, f.offset, parent);
            put32(file_meta, f.offset + 4, it + 1 != siblings.end() ? files[*(it + 1)].offset : kEmpty);
            put64(file_meta, f.offset + 8, f.data);
            put64(file_meta, f.offset + 16, f.size);
            put32(file_meta, f.offset + 24, bucket(file_hash, PathHash(parent, f.name) % file_buckets, f.offset));
            put32(file_meta, f.offset + 28, static_cast<u32>(f.name.size()));
            std::memcpy(file_meta.data() + f.offset + 0x20, f.name.data(), f.name.size());
        }

        meta_offset_ = Align(kDataOffset + data, 4);
        const u64 dir_hash_off = meta_offset_, dir_meta_off = dir_hash_off + dir_hash.size();
        const u64 file_hash_off = dir_meta_off + dir_meta.size(), file_meta_off = file_hash_off + file_hash.size();
        for (auto* t : {&dir_hash, &dir_meta, &file_hash, &file_meta}) {
            meta_.insert(meta_.end(), t->begin(), t->end());
        }
        size_ = meta_offset_ + meta_.size();
        head_.assign(kDataOffset, 0);
        const u64 header[10] = {0x50, dir_hash_off, dir_hash.size(), dir_meta_off, dir_meta.size(),
                                file_hash_off, file_hash.size(), file_meta_off, file_meta.size(), kDataOffset};
        std::memcpy(head_.data(), header, sizeof header);
        Log("romfs %s: %zu directories, %zu files, image 0x%llx bytes", roots[0].c_str(), dirs.size(), files.size(),
            (unsigned long long)size_);
        for (size_t l = 1; l < roots.size(); ++l) {
            Log("romfs: mod layer %s: %zu files replaced, %zu added", roots[l].c_str(), replaced[l], added[l]);
        }
    }

    u64 Size() const { return size_; }

    /// Copies [offset, offset+size) of the image into out (bytes past the end read as zero); the files read
    /// for the first time go to `opened` (their RomFS paths). Any thread, with or without the kernel lock.
    void Read(u64 offset, u8* out, u64 size, std::vector<std::string>& opened) {
        std::lock_guard lock(mu_);
        while (size) {
            u64 n;
            if (offset < kDataOffset) {
                n = std::min(size, kDataOffset - offset);
                std::memcpy(out, head_.data() + offset, n);
            } else if (offset >= meta_offset_) {
                const u64 rel = offset - meta_offset_;
                n = rel < meta_.size() ? std::min(size, meta_.size() - rel) : size;
                if (rel < meta_.size()) {
                    std::memcpy(out, meta_.data() + rel, n);
                } else {
                    std::memset(out, 0, n);
                }
            } else {
                // File data: the file containing offset, or alignment padding.
                auto it = std::upper_bound(files_.begin(), files_.end(), offset,
                                           [](u64 v, const HostFile& f) { return v < f.offset; });
                const u64 next = it == files_.end() ? meta_offset_ : it->offset;
                if (it != files_.begin() && offset < std::prev(it)->offset + std::prev(it)->size) {
                    HostFile& f = *std::prev(it);
                    n = std::min(size, f.offset + f.size - offset);
                    if (g_trace_romfs) {
                        Log("romfs read %s +0x%llx size 0x%llx", f.rel.c_str(),
                            (unsigned long long)(offset - f.offset), (unsigned long long)n);
                    }
                    if (!f.file.IsOpen()) {
                        opened.push_back(f.rel); // first read of this file
                        if (!f.file.Open(f.path, platform::File::Mode::kRead)) {
                            Fatal("romfs: cannot open %s", f.path.c_str());
                        }
                    }
                    if (f.file.ReadAt(out, n, offset - f.offset) != static_cast<s64>(n)) {
                        Fatal("romfs: read of %s failed", f.path.c_str());
                    }
                } else {
                    n = std::min(size, next - offset);
                    std::memset(out, 0, n);
                }
            }
            out += n;
            offset += n;
            size -= n;
        }
    }

private:
    struct HostFile {
        std::string path, rel; // the host file, and its path in the RomFS (as the host spells it)
        u64 offset, size; // in the image
        platform::File file; // opened on first read
    };
    std::mutex mu_; // Read: the files' lazy opens and positions
    std::vector<u8> head_, meta_;
    std::vector<HostFile> files_; // sorted by offset
    u64 meta_offset_ = 0, size_ = 0;
};

RomFs& TheRomFs() {
    static RomFs* romfs = [] {
        const char* dir = std::getenv("RT_ROMFS");
        if (!dir || !fs::is_directory(dir)) {
            Fatal("the game opened its RomFS: set RT_ROMFS to the extracted RomFS directory");
        }
        std::vector<std::string> roots{dir};
        for (auto& layer : mods::RomfsLayers()) roots.push_back(std::move(layer)); // RT_MODS (rt.cpp: already checked)
        return new RomFs(roots);
    }();
    return *romfs;
}

/// IStorage over the RomFS image (read-only).
class RomFsStorage : public Service {
public:
    RomFsStorage() : Service("fsp:IStorage(romfs)") {
        Register(0, "Read", [](IpcContext& ctx) {
            const u64 offset = ctx.Pop<u64>(), size = ctx.Pop<u64>();
            const IpcBuffer out = ctx.Out(0);
            if (size > out.size) {
                Fatal("IStorage::Read of 0x%llx bytes into a 0x%llx-byte buffer", (unsigned long long)size,
                      (unsigned long long)out.size);
            }
            const auto t0 = std::chrono::steady_clock::now();
            u8* dst = g_mem.Access(out.address, size, kWrite);
            std::vector<std::string> opened;
            if (g_romfs_locked) TheRomFs().Read(offset, dst, size, opened);
            else RunWithoutGil([&] { TheRomFs().Read(offset, dst, size, opened); });
            for (const std::string& name : opened) InputEvent(name);
            if (g_io_stats) {
                static u64 n = 0, bytes = 0;
                static double ms = 0, worst = 0;
                const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                bytes += size, ms += t, worst = std::max(worst, t);
                if (++n % 100 == 0) {
                    Log("romfs: %llu reads; last 100: %.1f MB in %.1f ms, slowest %.1f ms (t=%.2f s)", (unsigned long long)n,
                        bytes / 1048576.0, ms, worst, Uptime());
                    bytes = 0, ms = 0, worst = 0;
                }
            }
        });
        Register(4, "GetSize", [](IpcContext& ctx) { ctx.Push<u64>(TheRomFs().Size()); });
    }
};

class SdCardFileSystem : public Service {
public:
    SdCardFileSystem() : Service("fsp:IFileSystem(sdmc)") {}
};

// --- Host-backed IFileSystem (save data) ------------------------------------------
//
// A directory on the host stands for the file system. Request layouts and
// result codes follow drippu's fsp (fs_i_filesystem.cpp, fs_i_file.cpp,
// file_sys/errors.h, fs_directory.h).

constexpr u32 kResultPathNotFound = 0x202;      // FS 1
constexpr u32 kResultPathAlreadyExists = 0x402; // FS 2
constexpr u32 kOpenRead = 1, kOpenWrite = 2, kOpenAppend = 4;
constexpr u32 kDirFilterDirectories = 1, kDirFilterFiles = 2;

/// FileSys::DirectoryEntry.
struct DirectoryEntry {
    char name[0x301];
    u8 pad0[3];
    u8 type; // 0 directory, 1 file
    u8 pad1[3];
    s64 file_size;
};
static_assert(sizeof(DirectoryEntry) == 0x310);

class HostFile : public Service {
public:
    HostFile(platform::File file, std::string label) : Service("fsp:IFile(" + label + ")"), file_{std::move(file)} {
        Register(0, "Read", [this](IpcContext& ctx) {
            ctx.Pop<u32>(); // option
            ctx.Pop<u32>();
            const s64 offset = ctx.Pop<s64>(), size = ctx.Pop<s64>();
            const IpcBuffer out = ctx.Out(0);
            const u64 n = std::min<u64>(size, out.size);
            s64 got = 0;
            if (n) SaveIo("read", name, [&] { got = file_.ReadAt(g_mem.Access(out.address, n, kWrite), n, offset); });
            ctx.Push<s64>(got < 0 ? 0 : got);
        });
        Register(1, "Write", [this](IpcContext& ctx) {
            ctx.Pop<u32>(); // option (flush)
            ctx.Pop<u32>();
            const s64 offset = ctx.Pop<s64>(), size = ctx.Pop<s64>();
            const IpcBuffer in = ctx.In(0);
            const u64 n = std::min<u64>(size, in.size);
            s64 wrote = static_cast<s64>(n);
            if (n) SaveIo("write", name, [&] { wrote = file_.WriteAt(g_mem.Access(in.address, n, kRead), n, offset); });
            if (wrote != static_cast<s64>(n)) {
                Fatal("%s: host write of 0x%llx bytes failed", name.c_str(), (unsigned long long)n);
            }
        });
        Register(2, "Flush", [this](IpcContext&) { SaveIo("flush", name, [&] { file_.Flush(); }); });
        Register(3, "SetSize", [this](IpcContext& ctx) {
            const s64 size = ctx.Pop<s64>();
            bool ok = false;
            SaveIo("resize", name, [&] { ok = file_.SetSize(size); });
            if (!ok) Fatal("%s: SetSize failed", name.c_str());
        });
        Register(4, "GetSize", [this](IpcContext& ctx) { ctx.Push<s64>(file_.Size()); });
    }

private:
    platform::File file_;
};

class HostDirectory : public Service {
public:
    HostDirectory(const fs::path& dir, u32 filter) : Service("fsp:IDirectory") {
        for (const auto& e : fs::directory_iterator(dir)) {
            const bool is_dir = e.is_directory();
            if (!(filter & (is_dir ? kDirFilterDirectories : kDirFilterFiles))) continue;
            DirectoryEntry d{};
            const std::string n = e.path().filename().string();
            std::snprintf(d.name, sizeof d.name, "%s", n.c_str());
            d.type = is_dir ? 0 : 1;
            d.file_size = is_dir ? 0 : static_cast<s64>(e.file_size());
            entries_.push_back(d);
        }
        Register(0, "Read", [this](IpcContext& ctx) {
            const size_t room = ctx.Out(0).size / sizeof(DirectoryEntry);
            const size_t n = std::min(room, entries_.size() - next_);
            if (n) ctx.WriteOut(0, entries_.data() + next_, n * sizeof(DirectoryEntry));
            next_ += n;
            ctx.Push<s64>(static_cast<s64>(n));
        });
        Register(1, "GetEntryCount", [this](IpcContext& ctx) { ctx.Push<s64>(static_cast<s64>(entries_.size())); });
    }

private:
    std::vector<DirectoryEntry> entries_;
    size_t next_ = 0;
};

class HostFileSystem : public Service {
public:
    HostFileSystem(fs::path root, const std::string& label) : Service("fsp:IFileSystem(" + label + ")"), root_{std::move(root)} {
        fs::create_directories(root_);
        Register(0, "CreateFile", [this](IpcContext& ctx) {
            ctx.Pop<s32>(); // option
            const s64 size = ctx.Pop<s64>();
            const fs::path p = Host(ctx, 0);
            u32 result = 0;
            bool ok = true;
            SaveIo("create", name, [&] {
                if (fs::exists(p)) result = kResultPathAlreadyExists;
                else if (!fs::is_directory(p.parent_path())) result = kResultPathNotFound;
                else {
                    platform::File f;
                    ok = f.Open(p, platform::File::Mode::kCreate) && f.SetSize(size);
                }
            });
            if (result) return Fail(ctx, result);
            if (!ok) Fatal("%s: cannot create %s", name.c_str(), p.string().c_str());
        });
        Register(1, "DeleteFile", [this](IpcContext& ctx) {
            const fs::path p = Host(ctx, 0);
            Unlocked(ctx, "delete", [&]() -> u32 {
                if (!fs::is_regular_file(p)) return kResultPathNotFound;
                fs::remove(p);
                return 0;
            });
        });
        Register(2, "CreateDirectory", [this](IpcContext& ctx) {
            const fs::path p = Host(ctx, 0);
            Unlocked(ctx, "create directory", [&]() -> u32 {
                if (fs::exists(p)) return kResultPathAlreadyExists;
                if (!fs::is_directory(p.parent_path())) return kResultPathNotFound;
                fs::create_directory(p);
                return 0;
            });
        });
        Register(3, "DeleteDirectory", [this](IpcContext& ctx) {
            const fs::path p = Host(ctx, 0);
            Unlocked(ctx, "delete directory", [&]() -> u32 {
                if (!fs::is_directory(p)) return kResultPathNotFound;
                fs::remove(p);
                return 0;
            });
        });
        Register(4, "DeleteDirectoryRecursively", [this](IpcContext& ctx) {
            const fs::path p = Host(ctx, 0);
            Unlocked(ctx, "delete directory", [&]() -> u32 {
                if (!fs::is_directory(p)) return kResultPathNotFound;
                fs::remove_all(p);
                return 0;
            });
        });
        auto rename = [this](IpcContext& ctx) {
            const fs::path from = Host(ctx, 0), to = Host(ctx, 1);
            Unlocked(ctx, "rename", [&]() -> u32 {
                if (!fs::exists(from)) return kResultPathNotFound;
                if (fs::exists(to)) return kResultPathAlreadyExists;
                fs::rename(from, to);
                return 0;
            });
        };
        Register(5, "RenameFile", rename);
        Register(6, "RenameDirectory", rename);
        Register(7, "GetEntryType", [this](IpcContext& ctx) {
            const fs::path p = Host(ctx, 0);
            bool dir = false;
            if (Unlocked(ctx, "entry type", [&]() -> u32 {
                    if (!fs::exists(p)) return kResultPathNotFound;
                    dir = fs::is_directory(p);
                    return 0;
                })) {
                return;
            }
            ctx.Push<u32>(dir ? 0 : 1);
        });
        Register(8, "OpenFile", [this](IpcContext& ctx) {
            const u32 mode = ctx.Pop<u32>();
            const fs::path p = Host(ctx, 0);
            bool found = false, ok = false;
            platform::File f;
            SaveIo("open", name, [&] {
                found = fs::is_regular_file(p);
                ok = found && f.Open(p, mode & (kOpenWrite | kOpenAppend) ? platform::File::Mode::kReadWrite
                                                                          : platform::File::Mode::kRead);
            });
            if (!found) return Fail(ctx, kResultPathNotFound);
            if (!ok) Fatal("%s: cannot open %s", name.c_str(), p.string().c_str());
            ctx.PushObject(std::make_shared<HostFile>(std::move(f), p.filename().string()));
        });
        Register(9, "OpenDirectory", [this](IpcContext& ctx) {
            const u32 filter = ctx.Pop<u32>();
            const fs::path p = Host(ctx, 0);
            std::shared_ptr<HostDirectory> dir;
            if (Unlocked(ctx, "open directory", [&]() -> u32 {
                    if (!fs::is_directory(p)) return kResultPathNotFound;
                    dir = std::make_shared<HostDirectory>(p, filter); // reads the listing
                    return 0;
                })) {
                return;
            }
            ctx.PushObject(std::move(dir));
        });
        Stub(10, "Commit"); // writes go straight to the host files
        Register(11, "GetFreeSpaceSize", [](IpcContext& ctx) { ctx.Push<s64>(0x4000'0000); });
        Register(12, "GetTotalSpaceSize", [](IpcContext& ctx) { ctx.Push<s64>(0x4000'0000); });
        Register(13, "CleanDirectoryRecursively", [this](IpcContext& ctx) {
            const fs::path p = Host(ctx, 0);
            Unlocked(ctx, "clean directory", [&]() -> u32 {
                if (!fs::is_directory(p)) return kResultPathNotFound;
                for (const auto& e : fs::directory_iterator(p)) fs::remove_all(e.path());
                return 0;
            });
        });
        Register(14, "GetFileTimeStampRaw", [this](IpcContext& ctx) {
            const fs::path p = Host(ctx, 0);
            if (Unlocked(ctx, "time stamp", [&]() -> u32 { return fs::exists(p) ? 0 : kResultPathNotFound; })) return;
            const u64 stamp[4] = {}; // created, accessed, modified, padding: not tracked
            ctx.PushBytes(stamp, sizeof stamp);
        });
    }

private:
    static void Fail(IpcContext& ctx, u32 result) { ctx.result = result; }

    /// Runs `f` (the host part of a request: its result code, 0 for success) without the kernel lock (SaveIo);
    /// a non-zero result fails the request. Returns that result.
    template <typename F>
    u32 Unlocked(IpcContext& ctx, const char* what, F&& f) {
        u32 result = 0;
        SaveIo(what, name, [&] { result = f(); });
        if (result) Fail(ctx, result);
        return result;
    }

    /// The host path for the guest path in pointer buffer i ("/a/b"); ".." never leaves the root.
    fs::path Host(IpcContext& ctx, size_t i) const {
        const std::string guest = ctx.ReadInString(i);
        fs::path p = root_;
        for (const auto& part : fs::path(guest)) {
            const std::string s = part.string();
            if (s.empty() || s == "/" || s == ".") continue;
            if (s == "..") Fatal("%s: path %s leaves the file system", name.c_str(), guest.c_str());
            p /= part;
        }
        return p;
    }

    fs::path root_;
};

/// Save data lives in <RT_SAVE_DIR>/<program id>/<type>_<user id>/; RT_SAVE_DIR defaults to save/ under the working
/// directory (the bench scripts' run folders), and a player start sets it to the per-user folder's save/ (rt.cpp).
std::shared_ptr<HostFileSystem> OpenSaveData(IpcContext& ctx) {
    const u8 space_id = ctx.Pop<u8>();
    struct {
        u64 program_id;
        u64 user_id[2];
        u64 system_save_data_id;
        u8 type, rank;
        u16 index;
        u8 pad[0x1C];
    } attr;
    static_assert(sizeof attr == 0x40);
    attr = ctx.At<decltype(attr)>(8);
    const u64 program = attr.program_id ? attr.program_id : TitleNpdm().program_id;
    char dir[96];
    std::snprintf(dir, sizeof dir, "%016llx/%u_%016llx%016llx", (unsigned long long)program, attr.type,
                  (unsigned long long)attr.user_id[1], (unsigned long long)attr.user_id[0]);
    const char* root = std::getenv("RT_SAVE_DIR");
    const std::string path = (fs::path(root && *root ? root : "save") / dir).string();
    Log("fs   save data (space %u, index %u) -> %s", space_id, attr.index, path.c_str());
    std::shared_ptr<HostFileSystem> mounted;
    SaveIo("mount", path, [&] { mounted = std::make_shared<HostFileSystem>(path, "save"); }); // creates the folders
    return mounted;
}

class FspSrv : public Service {
public:
    FspSrv() : Service("fsp-srv") {
        Stub(1, "SetCurrentProcess");
        Register(18, "OpenSdCardFileSystem", [](IpcContext& ctx) { ctx.PushObject(std::make_shared<SdCardFileSystem>()); });
        Register(51, "OpenSaveDataFileSystem", [](IpcContext& ctx) { ctx.PushObject(OpenSaveData(ctx)); });
        Register(53, "OpenReadOnlySaveDataFileSystem", [](IpcContext& ctx) { ctx.PushObject(OpenSaveData(ctx)); });
        Register(200, "OpenDataStorageByCurrentProcess", [](IpcContext& ctx) {
            TheRomFs(); // build it now, so a missing RT_ROMFS fails here
            ctx.PushObject(std::make_shared<RomFsStorage>());
        });
        Register(1005, "GetGlobalAccessLogMode", [](IpcContext& ctx) { ctx.Push<u32>(0); }); // logging off
    }
};

const bool registered = [] {
    RegisterServiceFactory("fsp-srv", [] { return std::make_shared<FspSrv>(); });
    return true;
}();

} // namespace
} // namespace rt
