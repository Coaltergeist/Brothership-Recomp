// The player's options (options.h).
#include "options.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "rt.h"

namespace rt::options {

const std::vector<Option>& All() {
    static const std::vector<Option> all{
        {"RT_RES", "auto|360p|540p|720p|1080p|1440p|2160p|4k", "720p",
         "internal resolution (resolution.h; auto follows the window: 720p-1440p)"},
        {"RT_AA", "off|smaa", "off", "anti-aliasing of the game's image before it is scaled to the window (nvn/smaa.h)"},
        {"RT_WINDOW", "windowed|borderless", "windowed", "window mode (window.cpp; borderless fills the display)"},
        {"RT_WINDOW_SIZE", "<width>x<height>", "1280x720", "the window's size when windowed (window.cpp)"},
        {"RT_FPS", "30|60|uncapped", "the game's", "frame-rate cap (nvn/window.cpp; F8 cycles it in the window)"},
        {"RT_VSYNC", "mailbox|fifo|immediate", "mailbox", "present mode (nvn/display.cpp, gpu.cpp)"},
    };
    return all;
}

const char* Get(const char* name) {
    const Option* option = nullptr;
    for (const Option& o : All()) {
        if (std::strcmp(o.name, name) == 0) option = &o;
    }
    if (!option) Fatal("options: %s is not a player option (options.cpp All)", name);
    const char* v = std::getenv(name); // the launcher (B8): its settings file
    if (!v || !*v) return nullptr;
    if (!std::strchr(option->values, '<')) { // a fixed list: the value must be one of it
        const std::string values = std::string("|") + option->values + "|";
        if (values.find(std::string("|") + v + "|") == std::string::npos) {
            Fatal("%s=%s: expected %s", name, v, option->values);
        }
    }
    return v;
}

void LogAll() {
    std::string line;
    for (const Option& o : All()) {
        const char* v = Get(o.name);
        line += std::string(line.empty() ? "" : ", ") + o.name + "=" + (v ? v : std::string("(") + o.fallback + ")");
    }
    Log("options: %s", line.c_str());
}

} // namespace rt::options
