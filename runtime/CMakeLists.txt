cmake_minimum_required(VERSION 3.20)
project(nxrt C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

set(GEN_DIR "" CACHE PATH "Directory produced by nxrecomp -o")
if (NOT GEN_DIR)
    message(FATAL_ERROR "set -DGEN_DIR=<nxrecomp output directory>")
endif()

file(GLOB GEN_FUNCS "${GEN_DIR}/funcs_*.c")
# The lifted code (src/game/game.h): NXRT_GAME=static links it into nxrt; module builds nxrt without it -
# the game module (nxgame.dll / libnxgame.so) is built by tools/build_game.sh, freestanding, the way a player's
# builder builds it, and loaded at start. Both with src/game/entry.c, the names the lifted code calls and its one
# export, nx_game.
set(NXRT_GAME "static" CACHE STRING "The lifted code: static (in nxrt) or module (tools/build_game.sh builds it)")
set_property(CACHE NXRT_GAME PROPERTY STRINGS static module)
if (NXRT_GAME STREQUAL "module")
    add_library(nxgen INTERFACE) # (no game code here: the lift's headers only - GuestContext's layout)
    target_include_directories(nxgen INTERFACE "${GEN_DIR}")
elseif (NXRT_GAME STREQUAL "static")
    add_library(nxgen STATIC ${GEN_FUNCS} "${GEN_DIR}/nx_table.c" "${GEN_DIR}/recomp_runtime.c" src/game/entry.c)
else()
    message(FATAL_ERROR "NXRT_GAME=${NXRT_GAME}: static or module")
endif()
if (NXRT_GAME STREQUAL "static") # (to the endif below: the lifted code's compile options)
target_include_directories(nxgen PUBLIC "${GEN_DIR}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/game")
# Generated code: unused labels and variables are expected.
target_compile_options(nxgen PRIVATE -w)
# Optimisation for the generated code only, e.g. -O1 for a large program
# (appended after the build type's flags, so it wins).
set(NXGEN_OPT "" CACHE STRING "Extra compiler flags for the generated code")
if (NXGEN_OPT)
    separate_arguments(NXGEN_OPT_LIST UNIX_COMMAND "${NXGEN_OPT}")
    target_compile_options(nxgen PRIVATE ${NXGEN_OPT_LIST})
endif()
# The memory-access runtime (one file, called by every guest load and store) at -O2 even when the
# rest is built at -O1: with gcc its small helpers then inline (~5 % less time in the opening movie's
# decoder threads, 7 s of build); clang-cl measured unchanged. The FP helper shards
# at -O2 measured no faster (BUILD_TIME 3a).
if (MSVC)
    set(NXGEN_RUNTIME_OPT_DEFAULT "/clang:-O2")
else()
    set(NXGEN_RUNTIME_OPT_DEFAULT "-O2")
endif()
set(NXGEN_RUNTIME_OPT "${NXGEN_RUNTIME_OPT_DEFAULT}" CACHE STRING "Extra compiler flags for recomp_runtime.c (after NXGEN_OPT)")
if (NXGEN_RUNTIME_OPT)
    separate_arguments(NXGEN_RUNTIME_OPT_LIST UNIX_COMMAND "${NXGEN_RUNTIME_OPT}")
    set_source_files_properties("${GEN_DIR}/recomp_runtime.c" PROPERTIES COMPILE_OPTIONS "${NXGEN_RUNTIME_OPT_LIST}")
endif()
# Host ISA for the generated code: SSE4.2 on x86-64 (any CPU Windows 11 runs on has it; the lifted
# SIMD code uses pmovzx, pshufb, ptest). Empty for other hosts.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(NXGEN_ARCH_DEFAULT "-msse4.2")
else()
    set(NXGEN_ARCH_DEFAULT "")
endif()
set(NXGEN_ARCH "${NXGEN_ARCH_DEFAULT}" CACHE STRING "Target-ISA flags for the generated code")
if (NXGEN_ARCH)
    separate_arguments(NXGEN_ARCH_LIST UNIX_COMMAND "${NXGEN_ARCH}")
    target_compile_options(nxgen PRIVATE ${NXGEN_ARCH_LIST})
endif()
# count drippu's FP fast path (src/fpx_probe.h, force-included; its own build directory).
option(RT_FPX_PROBE "Count drippu's FP fast-path hits and misses (slower; for measurement)" OFF)
if (RT_FPX_PROBE)
    if (MSVC)
        target_compile_options(nxgen PRIVATE "/FI${CMAKE_CURRENT_SOURCE_DIR}/src/fpx_probe.h")
    else()
        target_compile_options(nxgen PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/src/fpx_probe.h")
    endif()
endif()
endif() # (NXRT_GAME static: the lifted code's compile options above)
# The generated code is meant for clang (clang-cl on Windows; on Linux configure a fresh build
# directory with -DCMAKE_C_COMPILER=$PWD/tools/llvm/bin/clang): GCC builds it correctly but turns
# its vector code into much slower sequences.
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    message(WARNING "The generated code is built with GCC: correct, but its SIMD is slower than with clang "
                    "(configure a fresh build directory with -DCMAKE_C_COMPILER=<clang>)")
endif()

# The host OS (src/platform/platform.h): one implementation per platform.
if (WIN32)
    set(RT_PLATFORM src/platform/windows.cpp)
else()
    set(RT_PLATFORM src/platform/posix.cpp)
endif()
file(GLOB RT_SERVICES "${CMAKE_CURRENT_SOURCE_DIR}/src/services/*.cpp")
find_package(Threads REQUIRED)

# Windows: what drippu's build defines there, for the runtime too (windows.h reaches it
# through drippu's Vulkan headers: no min/max macros, no rarely used headers).
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE _SCL_SECURE_NO_WARNINGS NOMINMAX
                            WIN32_LEAN_AND_MEAN UNICODE _UNICODE _WIN32_WINNT=0x0A00 WINVER=0x0A00)
    # A guest thread ends by throwing ThreadExit (threads.cpp) through rt_jump / rt_svc, which are extern "C":
    # /EHs, not CMake's default /EHsc, whose "c" lets the compiler assume extern "C" functions never throw
    # (the throw then ends the process).
    string(REPLACE "/EHsc" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
# drippu's commit (the shader store's tag, drippu's version string): from git, or given by a build from a source copy
# without .git (the Flatpak).
set(RT_DRIPPU_COMMIT "" CACHE STRING "drippu's commit, 12 digits (empty: from git)")
if (NOT RT_DRIPPU_COMMIT)
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/drippu rev-parse --short=12 HEAD
                    OUTPUT_VARIABLE RT_DRIPPU_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
# GPU and audio emulation (drippu's video_core and audio_core, built from third_party; see gpu/).
add_subdirectory(gpu)

# Window and (later) input and audio: SDL3, a static build from the submodule.
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
# Linux needs the X11 -dev packages SDL uses (README "Building the runtime").
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/SDL ${CMAKE_CURRENT_BINARY_DIR}/sdl EXCLUDE_FROM_ALL)

# The start screen's UI: Dear ImGui (the submodule, a release tag) with its SDL3 and Vulkan
# backends, and SDL's renderer for the shader cache's own run (ui/seed.h: no GPU API there); Vulkan's functions come from the NVN layer's device (IMGUI_IMPL_VULKAN_NO_PROTOTYPES, ui/start.cpp),
# against drippu's Vulkan headers.
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/imgui)
add_library(imgui STATIC ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_tables.cpp
                         ${IMGUI_DIR}/imgui_widgets.cpp ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
                         ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer3.cpp)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/drippu/externals/Vulkan-Headers/include)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_NO_PROTOTYPES IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
target_link_libraries(imgui PRIVATE SDL3::SDL3-static)

# Oodle decompression for the shader seed (src/seed/oodle.h): ooz, unmodified (its own warnings silenced); other
# systems than Windows get stand-ins for the Windows headers it includes.
set(OOZ_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ooz")
add_library(ooz STATIC src/seed/ooz.cpp ${OOZ_DIR}/lzna.cpp ${OOZ_DIR}/bitknit.cpp)
target_include_directories(ooz PRIVATE ${OOZ_DIR})
if (MSVC)
    target_compile_options(ooz PRIVATE /w)
else()
    target_include_directories(ooz PRIVATE src/seed/ooz_posix)
    # GCC at -O2 / -O3 miscompiles it (2026-11-01: 1,177 of the game's 56,113 IoStore chunks unpacked wrong on Linux,
    # 0 with clang-cl on Windows; -fno-strict-aliasing alone did not help, -O1 does: 0 wrong, the same 45,792 stages
    # as Epic's DLL). The extraction stays at ~11 s.
    target_compile_options(ooz PRIVATE -w -fno-strict-aliasing -O1)
endif()

add_executable(nxrt src/rt.cpp src/game/load.cpp src/build/build.cpp src/build/unpack.cpp src/seed/extract.cpp src/ui/start.cpp src/ui/seed.cpp src/writetrack.cpp src/svc.cpp src/threads.cpp src/kernel.cpp src/ipc.cpp src/capture.cpp src/gpu.cpp src/audio.cpp src/npdm.cpp src/perf.cpp src/diag.cpp src/window.cpp src/options.cpp src/mods.cpp src/resolution.cpp src/input.cpp src/fpx_probe.cpp src/overlay.cpp src/hle.cpp src/sequencer_fix.cpp src/nvn/tap.cpp
                    src/nvn/layer.cpp src/nvn/device.cpp src/nvn/state.cpp src/nvn/memory.cpp src/nvn/cmdbuf.cpp src/nvn/commands.cpp src/nvn/sampler.cpp src/nvn/texture.cpp src/nvn/program.cpp src/nvn/images.cpp src/nvn/shaders.cpp src/nvn/compute_translate.cpp src/nvn/rescaling_pass.cpp src/nvn/smaa.cpp src/nvn/shadertest.cpp src/nvn/window.cpp src/nvn/display.cpp src/nvn/render.cpp src/nvn/binds.cpp src/nvn/draws.cpp src/nvn/pipeline.cpp src/nvn/formats.cpp src/nvn/queue.cpp src/nvn/heavy.cpp src/nvn/shader_store.cpp src/nvn/research.cpp src/nvn/difftest.cpp
                    ${RT_SERVICES} ${RT_PLATFORM})
target_link_libraries(nxrt PRIVATE video_core audio_core imgui ooz SDL3::SDL3-static Threads::Threads)
if (NXRT_GAME STREQUAL "module")
    target_compile_definitions(nxrt PRIVATE NXRT_GAME_MODULE)
    target_link_libraries(nxrt PRIVATE nxgen) # (headers only: recomp_runtime.h, GuestContext's layout)
else()
    target_link_libraries(nxrt PRIVATE nxgen)
endif()
# The builder's module id (src/build/build.h): NXRT_TOOLCHAIN_ID, a hash of what a module is made with, regenerated
# when one of those files changes. A Linux source checkout's builder uses the repository's lifter and compiler.
set(RT_REPO "${CMAKE_CURRENT_SOURCE_DIR}/..")
file(GLOB_RECURSE RT_TOOLCHAIN_FILES CONFIGURE_DEPENDS RELATIVE "${RT_REPO}"
     "${RT_REPO}/lifter/src/*" "${RT_REPO}/third_party/drippu/src/core/recompiler/*" "${RT_REPO}/runtime/src/game/*")
list(APPEND RT_TOOLCHAIN_FILES runtime/src/build/flags.h)
list(SORT RT_TOOLCHAIN_FILES)
list(JOIN RT_TOOLCHAIN_FILES "\n" RT_TOOLCHAIN_LIST)
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/toolchain_files.txt" CONTENT "${RT_TOOLCHAIN_LIST}\n")
list(TRANSFORM RT_TOOLCHAIN_FILES PREPEND "${RT_REPO}/" OUTPUT_VARIABLE RT_TOOLCHAIN_PATHS)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gen_include/nxrt_toolchain_id.h"
                   COMMAND ${CMAKE_COMMAND} -DROOT=${RT_REPO} -DLIST=${CMAKE_CURRENT_BINARY_DIR}/toolchain_files.txt
                           -DOUT=${CMAKE_CURRENT_BINARY_DIR}/gen_include/nxrt_toolchain_id.h
                           -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain_id.cmake
                   COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/gen_include/nxrt_toolchain_id.h"
                   DEPENDS ${RT_TOOLCHAIN_PATHS} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain_id.cmake"
                   COMMENT "Hashing the game module's toolchain (NXRT_TOOLCHAIN_ID)" VERBATIM)
target_sources(nxrt PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen_include/nxrt_toolchain_id.h")
target_include_directories(nxrt PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen_include")
if (NOT WIN32)
    set_property(SOURCE src/build/build.cpp APPEND PROPERTY COMPILE_DEFINITIONS "NXRT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/..\"")
endif()
# drippu's compiled host shaders (nvn/smaa.cpp uses its SMAA shaders): the folder gpu/'s host_shaders makes
get_directory_property(RT_HOST_SHADERS_INCLUDE DIRECTORY gpu DEFINITION HS_OUT)
target_include_directories(nxrt PRIVATE ${RT_HOST_SHADERS_INCLUDE})
add_dependencies(nxrt host_shaders)
# the shader store's tag names drippu's commit (another translator: stored stages ignored).
set_property(SOURCE src/nvn/shader_store.cpp APPEND PROPERTY COMPILE_DEFINITIONS "RT_DRIPPU_COMMIT=\"${RT_DRIPPU_COMMIT}\"")
if (MSVC) # clang-cl: its -Wall means -Weverything; GCC's meaning through its GCC-style driver
    target_compile_options(nxrt PRIVATE /clang:-Wall /clang:-Wextra)
    # The executable is started from \\wsl.localhost: Windows copies it to local swap at start
    # instead of paging 1.6 GB of code over the network path while the game runs.
    target_link_options(nxrt PRIVATE /SWAPRUN:NET)
    # 16 MB stacks for every thread (Windows' default is 1 MB, Linux's 8 MB; lifted code nests deeply), as
    # LASRecomp does; winmm for timeBeginPeriod.
    target_link_options(nxrt PRIVATE /STACK:16777216)
    # A release has no console window: a Windows program still entered at main; with no terminal its
    # log goes to the per-user folder's nxrt.log (platform AttachOutput), its build jobs start without windows.
    option(NXRT_CONSOLE "Windows: a console program (off: no console window, the release)" ON)
    if (NOT NXRT_CONSOLE)
        target_link_options(nxrt PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
    endif()
    target_link_libraries(nxrt PRIVATE winmm)
else()
    target_compile_options(nxrt PRIVATE -Wall -Wextra)
endif()
# NVN functions keep the API's parameters whether they use them or not.
set_source_files_properties(src/nvn/layer.cpp src/nvn/device.cpp src/nvn/state.cpp src/nvn/memory.cpp src/nvn/cmdbuf.cpp src/nvn/commands.cpp src/nvn/sampler.cpp src/nvn/texture.cpp src/nvn/program.cpp src/nvn/images.cpp src/nvn/shaders.cpp src/nvn/compute_translate.cpp src/nvn/rescaling_pass.cpp src/nvn/smaa.cpp src/nvn/shadertest.cpp src/nvn/window.cpp src/nvn/display.cpp src/nvn/render.cpp src/nvn/binds.cpp src/nvn/draws.cpp src/nvn/pipeline.cpp src/nvn/formats.cpp src/nvn/queue.cpp src/nvn/heavy.cpp src/nvn/shader_store.cpp src/nvn/research.cpp src/nvn/difftest.cpp
                            PROPERTIES COMPILE_OPTIONS -Wno-unused-parameter)
