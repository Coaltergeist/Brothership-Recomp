# Brothership Recomp

A static recompilation of *Mario & Luigi: Brothership* (Nintendo Switch, version 1.0.1) into a native PC program,
in the spirit of [XenonRecomp](https://github.com/hedge-dev/XenonRecomp) and
[N64Recomp](https://github.com/N64Recomp/N64Recomp): the game's code is translated ahead of time into C and compiled
for the PC, with the Switch's system services and its NVN graphics API provided by our own runtime (graphics on
Vulkan). Windows first, the Steam Deck next.

## Status

**Beta preview (Windows).** The game runs on Windows (title, plaza, menus, battles; 60 fps, internal resolution up
to 4K, SMAA). The download is a launcher that installs the game from your own files: it unpacks your dump, translates
and compiles the game's code and prepares its shaders on your PC (about 10-15 minutes, ~25 GB free disk space while
it runs). The Steam Deck comes next.

## Installing

1. Download `BrothershipRecomp-Windows.zip` from the [releases](../../releases) and unzip it (not into Program Files).
2. Start `BrothershipRecomp.exe`.
3. **Settings > Game files:** the folder with your `.nsp` files (base game + 1.0.1 update).
   **Settings > Keys:** the folder with your `prod.keys`.
4. **Build**, then **Play**.

Settings, saves, the build and the caches live in `%LOCALAPPDATA%\brothership-recomp`. To report a problem,
attach `nxrt.log`, `build\build.log` and `build\unpack.log` from there (they never contain your keys).

## No game content

This repository contains **no Nintendo code, assets or keys**, and never will. To play you need your own copy of
the game (base game and the 1.0.1 update) dumped from your own Switch, and your own keys (`prod.keys`) from that
console. The game is translated and compiled on your machine; nothing of it is
distributed. Please do not ask for game files or keys.

## What is here

- `lifter/` - the recompiler: AArch64 machine code to C.
- `runtime/` - what the recompiled code runs on: memory, threads, the Switch's system services, the NVN graphics
  layer on Vulkan (shader translation, resolution scaling, anti-aliasing), sound and input.
- `tools/` - build, analysis and test tools (shader seed, cross-compilation, benchmarks).
- `tests/` - regression tests (a homebrew program run through the whole pipeline, platform tests).
- `third_party/` - libraries as git submodules (see [NOTICE.md](NOTICE.md)).

Building the launcher itself needs a developer setup (cross-compiled from WSL with clang; `tools/package_windows.sh`
makes the download).

## License

Copyright (C) 2026 The Brothership Recomp contributors.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
later version. See [LICENSE](LICENSE). Third-party code and its licences: [NOTICE.md](NOTICE.md).

Not affiliated with or endorsed by Nintendo. *Mario & Luigi* and *Nintendo Switch* are trademarks of Nintendo.
