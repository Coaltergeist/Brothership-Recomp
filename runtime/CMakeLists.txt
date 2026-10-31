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
add_library(nxgen STATIC ${GEN_FUNCS} "${GEN_DIR}/nx_table.c" "${GEN_DIR}/recomp_runtime.c")
target_include_directories(nxgen PUBLIC "${GEN_DIR}")
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
# decoder threads, 7 s of build); clang-cl measured unchanged (ARCHITECTURE 5.6). The FP helper shards
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
# SIMD code uses pmovzx, pshufb, ptest; SIMD_PLAN.md). Empty for other hosts.
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
# SIMD_PLAN Phase 4 step 0: count drippu's FP fast path (src/fpx_probe.h, force-included; its own build directory).
option(RT_FPX_PROBE "Count drippu's FP fast-path hits and misses (slower; for measurement)" OFF)
if (RT_FPX_PROBE)
    if (MSVC)
        target_compile_options(nxgen PRIVATE "/FI${CMAKE_CURRENT_SOURCE_DIR}/src/fpx_probe.h")
    else()
        target_compile_options(nxgen PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/src/fpx_probe.h")
    endif()
endif()
# The generated code is meant for clang (clang-cl on Windows; on Linux configure a fresh build
# directory with -DCMAKE_C_COMPILER=$PWD/tools/llvm/bin/clang): GCC builds it correctly but turns
# its vector code into much slower sequences (SIMD_PLAN.md §2).
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    message(WARNING "The generated code is built with GCC: correct, but its SIMD is slower than with clang "
                    "(configure a fresh build directory with -DCMAKE_C_COMPILER=<clang>; SIMD_PLAN.md)")
endif()

# The host OS (src/platform/platform.h): one implementation per platform.
if (WIN32)
    set(RT_PLATFORM src/platform/windows.cpp)
else()
    set(RT_PLATFORM src/platform/posix.cpp)
endif()
file(GLOB RT_SERVICES "${CMAKE_CURRENT_SOURCE_DIR}/src/services/*.cpp")
find_package(Threads REQUIRED)

# Windows (PORT_PLAN Phase 4): what drippu's build defines there, for the runtime too (windows.h reaches it
# through drippu's Vulkan headers: no min/max macros, no rarely used headers).
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE _SCL_SECURE_NO_WARNINGS NOMINMAX
                            WIN32_LEAN_AND_MEAN UNICODE _UNICODE _WIN32_WINNT=0x0A00 WINVER=0x0A00)
    # A guest thread ends by throwing ThreadExit (threads.cpp) through rt_jump / rt_svc, which are extern "C":
    # /EHs, not CMake's default /EHsc, whose "c" lets the compiler assume extern "C" functions never throw
    # (the throw then ends the process).
    string(REPLACE "/EHsc" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
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

add_executable(nxrt src/rt.cpp src/writetrack.cpp src/svc.cpp src/threads.cpp src/kernel.cpp src/ipc.cpp src/capture.cpp src/gpu.cpp src/audio.cpp src/npdm.cpp src/perf.cpp src/diag.cpp src/window.cpp src/options.cpp src/resolution.cpp src/input.cpp src/fpx_probe.cpp src/overlay.cpp src/hle.cpp src/nvn/tap.cpp
                    src/nvn/layer.cpp src/nvn/device.cpp src/nvn/state.cpp src/nvn/memory.cpp src/nvn/cmdbuf.cpp src/nvn/commands.cpp src/nvn/sampler.cpp src/nvn/texture.cpp src/nvn/program.cpp src/nvn/images.cpp src/nvn/shaders.cpp src/nvn/compute_translate.cpp src/nvn/rescaling_pass.cpp src/nvn/smaa.cpp src/nvn/shadertest.cpp src/nvn/window.cpp src/nvn/display.cpp src/nvn/render.cpp src/nvn/binds.cpp src/nvn/draws.cpp src/nvn/pipeline.cpp src/nvn/formats.cpp src/nvn/queue.cpp src/nvn/heavy.cpp src/nvn/shader_store.cpp src/nvn/research.cpp src/nvn/difftest.cpp
                    ${RT_SERVICES} ${RT_PLATFORM})
target_link_libraries(nxrt PRIVATE nxgen video_core audio_core SDL3::SDL3-static Threads::Threads)
# drippu's compiled host shaders (nvn/smaa.cpp uses its SMAA shaders): the folder gpu/'s host_shaders makes
get_directory_property(RT_HOST_SHADERS_INCLUDE DIRECTORY gpu DEFINITION HS_OUT)
target_include_directories(nxrt PRIVATE ${RT_HOST_SHADERS_INCLUDE})
add_dependencies(nxrt host_shaders)
# SHADER_PLAN Phase 4: the shader store's tag names drippu's commit (another translator: stored stages ignored).
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/drippu rev-parse --short=12 HEAD
                OUTPUT_VARIABLE RT_DRIPPU_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
set_property(SOURCE src/nvn/shader_store.cpp APPEND PROPERTY COMPILE_DEFINITIONS "RT_DRIPPU_COMMIT=\"${RT_DRIPPU_COMMIT}\"")
if (MSVC) # clang-cl: its -Wall means -Weverything; GCC's meaning through its GCC-style driver
    target_compile_options(nxrt PRIVATE /clang:-Wall /clang:-Wextra)
    # The executable is started from \\wsl.localhost (PORT_PLAN Phase 4): Windows copies it to local swap at start
    # instead of paging 1.6 GB of code over the network path while the game runs.
    target_link_options(nxrt PRIVATE /SWAPRUN:NET)
    # 16 MB stacks for every thread (Windows' default is 1 MB, Linux's 8 MB; lifted code nests deeply), as
    # LASRecomp does; winmm for timeBeginPeriod.
    target_link_options(nxrt PRIVATE /STACK:16777216)
    target_link_libraries(nxrt PRIVATE winmm)
else()
    target_compile_options(nxrt PRIVATE -Wall -Wextra)
endif()
# NVN functions keep the API's parameters whether they use them or not.
set_source_files_properties(src/nvn/layer.cpp src/nvn/device.cpp src/nvn/state.cpp src/nvn/memory.cpp src/nvn/cmdbuf.cpp src/nvn/commands.cpp src/nvn/sampler.cpp src/nvn/texture.cpp src/nvn/program.cpp src/nvn/images.cpp src/nvn/shaders.cpp src/nvn/compute_translate.cpp src/nvn/rescaling_pass.cpp src/nvn/smaa.cpp src/nvn/shadertest.cpp src/nvn/window.cpp src/nvn/display.cpp src/nvn/render.cpp src/nvn/binds.cpp src/nvn/draws.cpp src/nvn/pipeline.cpp src/nvn/formats.cpp src/nvn/queue.cpp src/nvn/heavy.cpp src/nvn/shader_store.cpp src/nvn/research.cpp src/nvn/difftest.cpp
                            PROPERTIES COMPILE_OPTIONS -Wno-unused-parameter)
