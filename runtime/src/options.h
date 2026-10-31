// The player's options (PORT_PLAN §6 B8): every setting the launcher will offer, in one table - its name, the
// values it takes, its default and what it does - and read only through Get(). Today Get() reads the environment
// variable of the same name; with the launcher it reads the launcher's settings file instead (this file is the one
// place to change). The run logs the chosen options in one line at start (LogAll: a bug report's settings).
//
// Controllers are the launcher's own (its button test, B8): no runtime option yet. Test and diagnostic switches
// (RT_TRACE_*, RT_NVN_*, ...) are not player options and stay environment variables.
#pragma once

#include <vector>

namespace rt::options {

struct Option {
    const char* name;     // also the environment variable
    const char* values;   // as the launcher offers them, '|'-separated
    const char* fallback; // the default (unset)
    const char* what;
};

/// The player's options, in the launcher's order.
const std::vector<Option>& All();

/// Player option `name`'s value, or nullptr when it is not set (an empty value counts as not set). A name not in
/// All() stops the run (every player option is in the table), and so does a value not in its list of values.
const char* Get(const char* name);

/// Logs every player option's value, or its default in parentheses, in one line.
void LogAll();

} // namespace rt::options
