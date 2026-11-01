// The player's options (PORT_PLAN §6 B8, LAUNCHER_PLAN): every setting the start screen offers, in one table - its
// name, the values it takes, its default and what it does - and read only through Get(). A value comes from the
// environment variable of the same name (tests, bench scripts, our play commands), else from the settings file the
// start screen writes (settings.ini in UserDir(): lines `RT_RES=1440p`; not read by test runs - headless or
// scripted input - unless RT_SETTINGS names it), else the default. The run logs the chosen
// options and where each came from in one line at start (LogAll: a bug report's settings).
//
// Controllers are the start screen's own (its button test, LAUNCHER_PLAN Phase 2): no runtime option yet. Test and
// diagnostic switches (RT_TRACE_*, RT_NVN_*, ...) are not player options and stay environment variables.
#pragma once

#include <string>
#include <vector>

namespace rt::options {

struct Option {
    const char* name;     // also the environment variable and the settings file's key
    const char* label;    // as the start screen names it
    const char* values;   // the values taken, '|'-separated; "<...>" = free text
    const char* names;    // each value as the start screen shows it, '|'-separated in the same order ("" = not
                          // offered there: an alias); nullptr = the values themselves
    const char* fallback; // the default (unset)
};

/// The player's options, in the start screen's order.
const std::vector<Option>& All();

/// Player option `name`'s value, or nullptr when it is not set (an empty value counts as not set). A name not in
/// All() stops the run (every player option is in the table), and so does an environment value not in its list of
/// values; a settings file value not in it is logged once and ignored.
const char* Get(const char* name);

/// Where Get(name)'s value comes from: "environment", "settings" or "default".
const char* Source(const char* name);

/// The settings file's value of `name` ("" = not set there), and setting it ("" removes it) - the start screen's.
std::string Saved(const char* name);
void Save(const char* name, const std::string& value);

/// Writes the settings file (UserDir() is made if missing); false, logged, when it cannot.
bool Write();

/// RT_DEBUG=on (Debug mode): the FPS box and the frame rate in the title, the function keys (F7 flag, F8 frame rate,
/// F9 log + capture, F10 toggle) and a controller's Capture button. Off (the default): none of them; the log is
/// written either way.
bool Debug();

/// The per-user folder: %LOCALAPPDATA%\brothership-recomp on Windows, $XDG_CONFIG_HOME (~/.config)/brothership-recomp
/// elsewhere. RT_SETTINGS=<file> names another settings file (tests).
std::string UserDir();
std::string SettingsPath();

/// Logs every player option's value, or its default in parentheses, and its source when not the default, in one
/// line.
void LogAll();

} // namespace rt::options
