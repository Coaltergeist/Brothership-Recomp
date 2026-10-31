#!/usr/bin/env python3
"""Case aliases for MSVC / Windows SDK files copied to a case-sensitive filesystem.

Windows headers and the sources that use them spell file names in any case
(`<Windows.h>`, `"WinSock2.h"`, `Kernel32.lib`); NTFS does not care, Linux does.
This script adds symbolic links so those spellings resolve:

  1. every file and directory under the given roots gets an all-lower-case alias;
  2. every `#include` in the roots' headers and in the extra source trees that
     names a file under an include root with a different case gets an alias with
     exactly that spelling;
  3. every library file gets aliases `<name>.lib` with the extension lower-case,
     in lower case and as written (`Kernel32.Lib` -> `Kernel32.lib`, `kernel32.lib`),
     and `#pragma comment(lib, "...")` spellings found in the scanned files.

Usage: winsdk_aliases.py <winsdk dir> [extra source dirs...]
Re-running is safe (existing names are left alone).
"""
import os
import re
import sys

INCLUDE_RE = re.compile(rb'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
PRAGMA_LIB_RE = re.compile(rb'#\s*pragma\s+comment\s*\(\s*lib\s*,\s*"([^"]+)"', re.I)
SOURCE_EXT = ('.h', '.hpp', '.hxx', '.inl', '.c', '.cc', '.cpp', '.cxx', '.idl', '.ipp')

made = 0


def link(target_name, alias_path):
    """alias_path -> target_name (relative, same directory)."""
    global made
    if os.path.lexists(alias_path):
        return
    os.symlink(target_name, alias_path)
    made += 1


def lower_aliases(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            if name != name.lower():
                link(name, os.path.join(dirpath, name.lower()))


def resolve_ci(base, rel):
    """The real path of `rel` under `base`, matching each component case-insensitively; None if absent."""
    cur = base
    for part in rel.replace('\\', '/').split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            cur = os.path.dirname(cur)
            continue
        if not os.path.isdir(cur):
            return None
        exact = os.path.join(cur, part)
        if os.path.lexists(exact):
            cur = exact
            continue
        hits = [n for n in os.listdir(cur) if n.lower() == part.lower()]
        if not hits:
            return None
        cur = os.path.join(cur, hits[0])
    return cur


def spelled_alias(base, rel):
    """Make `rel` (as spelled) resolve under `base` by adding aliases component by component."""
    cur = base
    for part in rel.replace('\\', '/').split('/'):
        if part in ('', '.', '..'):
            return  # (relative climbs: left alone)
        exact = os.path.join(cur, part)
        if not os.path.lexists(exact):
            hits = [n for n in os.listdir(cur) if n.lower() == part.lower() and not os.path.islink(os.path.join(cur, n))]
            if not hits:
                hits = [n for n in os.listdir(cur) if n.lower() == part.lower()]
            if not hits:
                return
            link(hits[0], exact)
        cur = exact
        if not os.path.isdir(cur):
            return


def scan(paths, include_roots, lib_dirs):
    for top in paths:
        for dirpath, _, filenames in os.walk(top):
            for name in filenames:
                if not name.lower().endswith(SOURCE_EXT):
                    continue
                try:
                    data = open(os.path.join(dirpath, name), 'rb').read()
                except OSError:
                    continue
                for m in INCLUDE_RE.finditer(data):
                    rel = m.group(1).decode('latin1')
                    for base in include_roots:
                        if resolve_ci(base, rel) and not os.path.exists(os.path.join(base, rel)):
                            spelled_alias(base, rel)
                for m in PRAGMA_LIB_RE.finditer(data):
                    lib = m.group(1).decode('latin1')
                    if not lib.lower().endswith('.lib'):
                        lib += '.lib'
                    for d in lib_dirs:
                        hits = [n for n in os.listdir(d) if n.lower() == lib.lower() and not os.path.islink(os.path.join(d, n))]
                        if hits:
                            link(hits[0], os.path.join(d, lib))


def lib_aliases(lib_dirs):
    for d in lib_dirs:
        for name in os.listdir(d):
            if os.path.islink(os.path.join(d, name)) or not name.lower().endswith('.lib'):
                continue
            stem = name[:-4]
            for alias in (stem + '.lib', stem.lower() + '.lib'):
                if alias != name:
                    link(name, os.path.join(d, alias))


def main():
    w = os.path.abspath(sys.argv[1])
    include_roots = [os.path.join(w, 'msvc', 'include')] + [
        os.path.join(w, 'sdk', 'include', d) for d in ('ucrt', 'um', 'shared', 'winrt')]
    lib_dirs = [os.path.join(w, 'msvc', 'lib', 'x64'), os.path.join(w, 'sdk', 'lib', 'ucrt', 'x64'),
                os.path.join(w, 'sdk', 'lib', 'um', 'x64')]
    for r in include_roots:
        lower_aliases(r)
    lib_aliases(lib_dirs)
    scan(include_roots + [os.path.abspath(p) for p in sys.argv[2:]], include_roots, lib_dirs)
    print(f'{made} aliases added')


if __name__ == '__main__':
    main()
