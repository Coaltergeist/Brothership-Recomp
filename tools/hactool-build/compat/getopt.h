/* getopt_long for Microsoft's C library (it has none): only what hactool's main.c uses - options before the file
   names (no reordering), --name=value or --name value, -x and -xvalue / -x value. Ours (GPL-3.0-or-later). */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct option {
    const char *name;
    int has_arg; /* no_argument, required_argument, optional_argument */
    int *flag;
    int val;
};
#define no_argument 0
#define required_argument 1
#define optional_argument 2

extern char *optarg;
extern int optind, opterr, optopt;

int getopt_long(int argc, char *const argv[], const char *shortopts, const struct option *longopts, int *longindex);

#ifdef __cplusplus
}
#endif
