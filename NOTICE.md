# Notices

Brothership Recomp: Copyright (C) 2026 The Brothership Recomp contributors, licensed under the GNU General Public
License version 3 or later ([LICENSE](LICENSE)).

## Libraries (git submodules in `third_party/`, each with its own licence files)

| Library | Licence | Used for |
|---|---|---|
| [drippu](https://github.com/suyu-emu/drippu) (suyu; from yuzu) | GPL-3.0-or-later (yuzu's parts GPL-2.0-or-later) | the shader recompiler, Vulkan helpers, audio |
| [fmt](https://github.com/fmtlib/fmt) | MIT-style (fmt licence) | formatting (drippu) |
| [glslang](https://github.com/KhronosGroup/glslang) | BSD-3-Clause and others (its LICENSE.txt) | shader compilation at build time |
| [Boost](https://github.com/boostorg/boost) | BSL-1.0 | drippu's dependencies |
| [SDL](https://github.com/libsdl-org/SDL) | zlib | window, input, sound |
| [Dear ImGui](https://github.com/ocornut/imgui) | MIT | the start screen |
| [ooz](https://github.com/powzix/ooz) (Powzix) | GPL-3.0-or-later (its source files' headers) | Oodle decompression of the game's archives, for the shader seed built on the player's PC |
| [hactool](https://github.com/SciresM/hactool) (SciresM) | ISC | unpacking the player's own .nsp files with their own keys (shipped as `tools/hactool.exe`) |
| [mbed TLS](https://github.com/Mbed-TLS/mbedtls) (in hactool) | Apache-2.0 | hactool's cryptography |

## Code adapted from other projects

- **The 60FPS mod for Mario & Luigi: Brothership** by Fl4sh_#9174 ([github.com/Fl4sh9174](https://github.com/Fl4sh9174),
  [ko-fi.com/Fl4sh9174](https://Ko-fi.com/Fl4sh9174)): the frame-rate patch built into the game (the Frame rate
  setting switches it). Thanks to the author.

- **drippu** (GPL-3.0-or-later / GPL-2.0-or-later): `runtime/src/nvn/rescaling_pass.cpp` is adapted from its
  `shader_recompiler/ir_opt/rescaling_pass.cpp`; `runtime/src/nvn/smaa.cpp` from its
  `video_core/renderer_vulkan/present/smaa.cpp` (and uses its SMAA shaders).
- **LASRecomp** ([peiikai/LASRecomp](https://github.com/peiikai/LASRecomp), MIT): the NVN-to-Vulkan format table in
  `runtime/src/nvn/images.cpp` follows its `textures.c`. Its licence:

```
MIT License

Copyright (c) 2026 Peiikai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This license covers LAS Recomp's own code. It does not cover The Legend of Zelda: Link's Awakening or any other
Nintendo property, nor the third-party components in third_party/, which keep their own licenses.
```

Other recompilation projects (XenonRecomp, N64Recomp, the SSBU recomp, SkyRecomp, nxnative) were studied for their
designs; no code was taken from them except as listed above.
