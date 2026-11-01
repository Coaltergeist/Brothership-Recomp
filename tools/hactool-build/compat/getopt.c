/* getopt_long for Microsoft's C library (getopt.h). */
#include "getopt.h"

#include <stdio.h>
#include <string.h>

char *optarg = 0;
int optind = 1, opterr = 1, optopt = 0;
static const char *next_short = 0; /* the rest of a -abc group */

int getopt_long(int argc, char *const argv[], const char *shortopts, const struct option *longopts, int *longindex) {
    optarg = 0;
    if (!next_short || !*next_short) {
        next_short = 0;
        if (optind >= argc || argv[optind][0] != '-' || !argv[optind][1]) return -1;
        if (!strcmp(argv[optind], "--")) return ++optind, -1;
        if (argv[optind][1] == '-') { /* --name[=value] */
            const char *name = argv[optind] + 2, *eq = strchr(name, '=');
            const size_t len = eq ? (size_t)(eq - name) : strlen(name);
            ++optind;
            for (int i = 0; longopts && longopts[i].name; ++i) {
                if (strlen(longopts[i].name) != len || strncmp(longopts[i].name, name, len)) continue;
                if (longindex) *longindex = i;
                if (longopts[i].has_arg == required_argument) {
                    if (eq) optarg = (char *)eq + 1;
                    else if (optind < argc) optarg = argv[optind++];
                    else {
                        if (opterr) fprintf(stderr, "option --%s needs a value\n", longopts[i].name);
                        return '?';
                    }
                } else if (longopts[i].has_arg == optional_argument && eq) {
                    optarg = (char *)eq + 1;
                }
                if (longopts[i].flag) return *longopts[i].flag = longopts[i].val, 0;
                return longopts[i].val;
            }
            if (opterr) fprintf(stderr, "unknown option --%.*s\n", (int)len, name);
            return '?';
        }
        next_short = argv[optind++] + 1;
    }
    const char c = *next_short++;
    const char *spec = strchr(shortopts, c);
    if (!spec || c == ':') {
        optopt = c;
        if (opterr) fprintf(stderr, "unknown option -%c\n", c);
        return '?';
    }
    if (spec[1] == ':') {
        if (*next_short) optarg = (char *)next_short;
        else if (optind < argc) optarg = argv[optind++];
        else {
            optopt = c;
            return '?';
        }
        next_short = 0;
    }
    return c;
}
