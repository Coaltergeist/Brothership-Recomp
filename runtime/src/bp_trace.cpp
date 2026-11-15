// Brothership's Blueprint calls, logged by name (2026-11-10): which script functions the game runs, and when.
//
// Much of the game's field logic is Blueprint: behaviour-tree services, gimmick objects, field events. A bug that
// shows only at 60 fps there leaves nothing in our log, and the scripts' code (a Blueprint disassembler reads it from the
// unpacked assets) says what can happen, not what did. This logs each call a script makes to a function, with the
// frame, the object and its class, so a 30 fps run and a 60 fps run of the same moment can be compared line by line.
//
// RT_BP_TRACE=<text>[,<text>...]: the calls whose object's class or function contains one of the texts (case
// ignored), e.g. RT_BP_TRACE=Seesaw,FieldEvent; RT_BP_TRACE=all: every call (thousands a frame). Lines: `bp: f<frame>
// <object> <object's class>: <function's class>::<function>`, a call of an event graph with its entry offset (the offsets
// a disassembly of the graph shows). RT_BP_WATCH=<property>[,...]: each logged line also shows these properties of its object
// (integers, bytes, array counts, vectors), read before the call runs; `A.B` shows property B of the object in A,
// and so on (RootComponent.RelativeLocation: an actor's position); a last step `@<hex>` shows the 3 floats at that
// offset of the object, for native fields with no property (BattleCharaBase.@228: a battle character's home spot). Calls from native code into a script (events, ticks) are not seen, the calls those make are.
//
// How: the interpreter (UE 4.27 ScriptCore.cpp) runs each opcode through its handler table, so the four call
// opcodes' handlers are reached through pointers and can be overridden (hle.h). Each gets the context object in x0
// and the FFrame in x1, whose Code pointer (+0x20) is at the opcode's operand: the UFunction* (final calls) or the
// function's name (virtual calls: comparison index, display index, number). Names come from the name pool, found
// once by its first entries ("None", "ByteProperty") behind a pointer in main's data.
//
// Game 1.0.1's code (guest = main ELF + 0x8004000), checked word by word before anything is installed:
//   0x8d921f8  UObject::execVirtualFunction       0x8d92250  UObject::execFinalFunction
//   0x8d92260  UObject::execLocalVirtualFunction  0x8d922e8  UObject::execLocalFinalFunction
#include "bp_trace.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "hle.h"
#include "perf.h"
#include "rt.h"

namespace rt::bp_trace {
namespace {

struct Handler {
    u64 va;
    u32 words[4];
    bool by_name; // the operand is the function's name, not the UFunction
};
constexpr Handler kHandlers[] = {
    {0x8d921f8, {0xa9bd7bfd, 0xa9024ff4, 0xf9401028, 0xaa0103f4}, true},
    {0x8d92250, {0xf9401028, 0xf8408503, 0xf9001028, 0x17ffe309}, false},
    {0x8d92260, {0xa9bd7bfd, 0xa9024ff4, 0xf9401028, 0xaa0103f4}, true},
    {0x8d922e8, {0xaa0203e3, 0xaa0103e2, 0xf9401049, 0xf8408528}, false},
};
// main's data and bss (ELF 0x5b03000, 0x1b90000 bytes), where the name pool lies
constexpr u64 kDataBegin = 0x8004000 + 0x5b03000, kDataEnd = kDataBegin + 0x1b90000;
// UObject: class +0x10, name +0x18 (comparison index, number), outer +0x20; FFrame: Code +0x20
constexpr u64 kClass = 0x10, kName = 0x18, kOuter = 0x20, kCode = 0x20;

std::vector<std::string> g_filters; // lower case; empty: all
std::mutex g_lock;
u64 g_blocks = 0; // the name pool's block pointers; 1: not found
std::unordered_map<u64, std::string> g_names;

bool Peek(u64 va, void* out, u64 size) {
    const Region* r = g_mem.Find(va);
    if (!r || va + size > r->base + r->size) return false;
    std::memcpy(out, g_mem.Raw(va), size);
    return true;
}
template <typename T>
T Get(u64 va) {
    T v{};
    Peek(va, &v, sizeof v);
    return v;
}

// A name entry: a 16-bit header (wide: bit 0, length: bits 6-15), then the characters.
bool Entry(u64 va, std::string* out) {
    u16 header;
    if (!Peek(va, &header, 2)) return false;
    const u32 len = header >> 6;
    out->clear();
    if (header & 1) {
        for (u32 i = 0; i < len; ++i) {
            const u16 ch = Get<u16>(va + 2 + 2 * i);
            out->push_back(ch < 0x80 ? static_cast<char>(ch) : '?');
        }
        return true;
    }
    out->resize(len);
    return len == 0 || Peek(va + 2, out->data(), len);
}

void FindPool() {
    std::string first, second;
    for (u64 at = kDataBegin; at + 8 <= kDataEnd; at += 8) {
        u64 block;
        std::memcpy(&block, g_mem.Raw(at), 8);
        if (block < 0x10000 || (block & 1)) continue;
        if (Entry(block, &first) && first == "None" && Entry(block + 6, &second) && second == "ByteProperty") {
            g_blocks = at;
            Log("bp trace: the name pool's blocks at 0x%llx", (unsigned long long)at);
            return;
        }
    }
    g_blocks = 1;
    Log("bp trace: name pool not found: names print as #<index>");
}

std::string Name(u64 va) { // the FName at va
    const u32 index = Get<u32>(va), number = Get<u32>(va + 4);
    const u64 key = static_cast<u64>(number) << 32 | index;
    std::lock_guard lock(g_lock);
    if (auto it = g_names.find(key); it != g_names.end()) return it->second;
    if (g_blocks == 0) FindPool();
    std::string s;
    const u64 block = g_blocks > 1 ? Get<u64>(g_blocks + 8 * (index >> 16)) : 0;
    if (!block || !Entry(block + 2 * (index & 0xffff), &s)) s = "#" + std::to_string(index);
    if (number) s += "_" + std::to_string(number - 1);
    g_names.emplace(key, s);
    return s;
}

std::string NameAt(u64 code) { // a script name operand: comparison index, display index, number
    u32 pair[2] = {Get<u32>(code), Get<u32>(code + 8)};
    u64 key = static_cast<u64>(pair[1]) << 32 | pair[0];
    {
        std::lock_guard lock(g_lock);
        if (auto it = g_names.find(key); it != g_names.end()) return it->second;
    }
    // (Name reads an FName in guest memory: resolve through the same path with the pair in hand)
    std::lock_guard lock(g_lock);
    if (g_blocks == 0) FindPool();
    std::string s;
    const u64 block = g_blocks > 1 ? Get<u64>(g_blocks + 8 * (pair[0] >> 16)) : 0;
    if (!block || !Entry(block + 2 * (pair[0] & 0xffff), &s)) s = "#" + std::to_string(pair[0]);
    if (pair[1]) s += "_" + std::to_string(pair[1] - 1);
    g_names.emplace(key, s);
    return s;
}

// RT_BP_WATCH: the watched properties of a class (its own and its parents'), found once per class by name.
// UStruct: super +0x40, first property +0x50; FField (0x34 bytes): class +0x08, next +0x20, name +0x28; FFieldClass:
// name +0x00; FProperty: element size +0x38, offset +0x4c; FBoolProperty: field mask +0x7b; FArrayProperty: inner
// +0x78. A name with dots (RootComponent.RelativeLocation) follows object properties to the last one (up to 4 levels),
// e.g. an actor's position, which lives on its root component.
struct Watch {
    std::string name, type, inner; // type: the property class's name (IntProperty, ArrayProperty, ...)
    u32 offset, size;
    u8 mask;
    std::string rest; // after the dot: the property of the object this one points to
};
std::vector<std::string> g_watch;
std::unordered_map<u64, std::vector<Watch>> g_watches; // by class; under g_watch_lock
std::mutex g_watch_lock;

bool FindPropertyUncached(u64 cls, const std::string& name, Watch* out);

// The property `name` of a class (its own or a parent's), as a Watch; false if it has none. Cached per class and name
// (under g_watch_lock), so a path watched on every frame costs a lookup, not a walk of the class's properties.
bool FindProperty(u64 cls, const std::string& name, Watch* out) {
    static std::unordered_map<std::string, std::pair<bool, Watch>> cache;
    const std::string key = std::to_string(cls) + "/" + name;
    auto it = cache.find(key);
    if (it == cache.end()) {
        Watch w;
        const bool found = FindPropertyUncached(cls, name, &w);
        it = cache.emplace(key, std::make_pair(found, std::move(w))).first;
    }
    *out = it->second.second;
    return it->second.first;
}

bool FindPropertyUncached(u64 cls, const std::string& name, Watch* out) {
    int structs = 0;
    for (u64 s = cls; s && structs < 64; s = Get<u64>(s + 0x40), ++structs) {
        int fields = 0;
        for (u64 f = Get<u64>(s + 0x50); f && fields < 4096; f = Get<u64>(f + 0x20), ++fields) {
            if (Name(f + 0x28) != name) continue;
            *out = Watch{name, Name(Get<u64>(f + 0x08)), "", Get<u32>(f + 0x4c), Get<u32>(f + 0x38),
                         Get<u8>(f + 0x7b), ""};
            if (out->type == "ArrayProperty") {
                const u64 inner = Get<u64>(f + 0x78);
                out->inner = Name(Get<u64>(inner + 0x08));
                out->size = Get<u32>(inner + 0x38);
                out->mask = Get<u8>(inner + 0x7b);
            }
            return true;
        }
    }
    return false;
}

std::string Element(const std::string& type, u64 at, u32 size, u8 mask) {
    char text[96];
    if (type == "ObjectProperty" || type == "WeakObjectProperty" || type == "ClassProperty") {
        const u64 object = type == "WeakObjectProperty" ? 0 : Get<u64>(at);
        return object ? Name(object + kName) : (type == "WeakObjectProperty" ? "weak" : "none");
    }
    if (type == "NameProperty") return Name(at);
    if (type == "BoolProperty") return (Get<u8>(at) & mask) ? "true" : "false";
    if (type == "IntProperty") std::snprintf(text, sizeof text, "%d", Get<s32>(at));
    else if (type == "FloatProperty") std::snprintf(text, sizeof text, "%g", Get<float>(at));
    else if (type == "ByteProperty" || type == "EnumProperty") std::snprintf(text, sizeof text, "%u", Get<u8>(at));
    else if (type == "StructProperty" && size == 12)
        std::snprintf(text, sizeof text, "(%g, %g, %g)", Get<float>(at), Get<float>(at + 4), Get<float>(at + 8));
    else std::snprintf(text, sizeof text, "<%s %u bytes>", type.c_str(), size);
    return text;
}

std::string Watched(u64 object, u64 cls) {
    if (g_watch.empty()) return {};
    std::lock_guard lock(g_watch_lock);
    auto it = g_watches.find(cls);
    if (it == g_watches.end()) {
        std::vector<Watch> found;
        for (const std::string& path : g_watch) {
            const size_t dot = path.find('.');
            Watch w;
            if (!FindProperty(cls, path.substr(0, dot), &w)) continue;
            if (dot != std::string::npos) {
                if (w.type != "ObjectProperty") continue;
                w.rest = path.substr(dot + 1);
                w.name = path;
            }
            found.push_back(std::move(w));
        }
        it = g_watches.emplace(cls, std::move(found)).first;
    }
    std::string out;
    for (const Watch& w : it->second) {
        const u64 at = object + w.offset;
        out += " " + w.name + "=";
        if (!w.rest.empty()) { // through the objects the path's properties point to
            u64 sub = Get<u64>(at);
            std::string rest = w.rest;
            Watch v;
            bool ok = false;
            for (int depth = 0; sub && depth < 4; ++depth) {
                const size_t dot = rest.find('.');
                if (rest[0] == '@' && dot == std::string::npos) { // a native field: 3 floats at that offset
                    v = Watch{rest, "StructProperty", "", 0, 12, 0, ""};
                    sub += std::strtoull(rest.c_str() + 1, nullptr, 16);
                    ok = true;
                    break;
                }
                if (!FindProperty(Get<u64>(sub + kClass), rest.substr(0, dot), &v)) break;
                if (dot == std::string::npos) {
                    ok = v.type != "ArrayProperty";
                    break;
                }
                if (v.type != "ObjectProperty") break;
                sub = Get<u64>(sub + v.offset);
                rest = rest.substr(dot + 1);
            }
            out += ok ? Element(v.type, sub + v.offset, v.size, v.mask) : (sub ? "?" : "none");
            continue;
        }
        if (w.type != "ArrayProperty") {
            out += Element(w.type, at, w.size, w.mask);
            continue;
        }
        const u64 data = Get<u64>(at);
        const s32 count = Get<s32>(at + 8);
        out += "[";
        for (s32 i = 0; i < count && i < 8; ++i) {
            out += (i ? ", " : "") + Element(w.inner, data + static_cast<u64>(i) * w.size, w.size, w.mask);
        }
        out += count > 8 ? ", ...]" : "]";
    }
    return out;
}

template <typename T>
T Fast(u64 va) { // unchecked: the interpreter reads these itself right after
    T v;
    std::memcpy(&v, g_mem.Raw(va), sizeof v);
    return v;
}

bool Wanted(const std::string& text) {
    if (g_filters.empty()) return true;
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return std::tolower(ch); });
    for (const std::string& f : g_filters) {
        if (lower.find(f) != std::string::npos) return true;
    }
    return false;
}

void Call(GuestContext* c, void* user) {
    const Handler& h = *static_cast<const Handler*>(user);
    const u64 object = c->x[0], code = Fast<u64>(c->x[1] + kCode), cls = Fast<u64>(object + kClass);
    // The filter looks at the object's class and the function, decided once per pair (and thread): a call that
    // is not wanted costs three reads and a lookup, so the trace does not change the game's frame times much
    // (a first version that built every line turned a 60 fps failure into a pass).
    const u64 function = h.by_name ? (1ull << 63 | static_cast<u64>(Fast<u32>(code + 8)) << 32 | Fast<u32>(code))
                                   : Fast<u64>(code);
    thread_local std::unordered_map<u64, std::pair<bool, std::string>> t_known;
    auto it = t_known.find(cls * 0x9E3779B97F4A7C15ull ^ function);
    if (it == t_known.end()) {
        std::string text = Name(cls + kName) + ": ";
        text += h.by_name ? NameAt(code) : Name(Get<u64>(function + kOuter) + kName) + "::" + Name(function + kName);
        const bool wanted = Wanted(text);
        it = t_known.emplace(cls * 0x9E3779B97F4A7C15ull ^ function, std::make_pair(wanted, std::move(text))).first;
    }
    if (it->second.first) {
        const u64 after = code + (h.by_name ? 12 : 8);
        char entry[24] = "";
        if (Get<u8>(after) == 0x1d) std::snprintf(entry, sizeof entry, "(%d)", Get<s32>(after + 1)); // a graph's entry
        Log("bp: f%llu %s %s%s%s", (unsigned long long)perf::Frames(), Name(object + kName).c_str(),
            it->second.second.c_str(), entry, Watched(object, cls).c_str());
    }
    hle::CallOriginal(c, h.va);
}

} // namespace

void Install() {
    const char* list = std::getenv("RT_BP_TRACE");
    if (!list || !*list) return;
    for (const Handler& h : kHandlers) {
        u32 words[4] = {};
        if (!Peek(h.va, words, sizeof words) || std::memcmp(words, h.words, sizeof words) != 0) {
            Log("bp trace: off (the game's code at 0x%llx is not 1.0.1's)", (unsigned long long)h.va);
            return;
        }
    }
    std::string item;
    for (const char* p = list;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!item.empty() && item != "all") g_filters.push_back(item);
            item.clear();
            if (*p == '\0') break;
        } else {
            item.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
        }
    }
    if (const char* watch = std::getenv("RT_BP_WATCH")) { // (names as the scripts spell them, case and spaces kept)
        std::string name;
        for (const char* p = watch;; ++p) {
            if (*p == ',' || *p == '\0') {
                if (!name.empty()) g_watch.push_back(name);
                name.clear();
                if (*p == '\0') break;
            } else {
                name.push_back(*p);
            }
        }
    }
    for (const Handler& h : kHandlers) {
        char name[32];
        std::snprintf(name, sizeof name, "0x%llx", (unsigned long long)h.va);
        hle::OverrideGuestFunction(name, Call, const_cast<Handler*>(&h));
    }
    Log("bp trace: on (RT_BP_TRACE=%s): script function calls logged as `bp: f<frame> <object> <class>: <function>`", list);
}

} // namespace rt::bp_trace
