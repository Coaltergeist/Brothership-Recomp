# GPU emulation reused from drippu (yuzu family): a subset of `common`,
# `shader_recompiler` and `video_core`, built from the third_party submodules.
# drippu's `core/` is replaced by the headers in shim/ (they come first in the
# include path) and by the runtime's implementations of them.
#
# drippu's own top-level CMake (CPM, the whole emulator) is not used. Where
# its CMakeLists can be included as they are, they are; video_core's and the
# host shaders' source lists are read from drippu's files at configure time.

# Also configurable on its own (cmake -S runtime/gpu), to build the libraries
# without the runtime and its generated code.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(nxgpu C CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    find_package(Threads REQUIRED)
endif()

set(TP ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)
set(DRIPPU ${TP}/drippu)
set(DRIPPU_SRC ${DRIPPU}/src)
set(DRIPPU_EXT ${DRIPPU}/externals)

# Variables and functions drippu's CMakeLists read.
set(ARCHITECTURE_x86_64 ON)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CXX_GCC ON)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CXX_CLANG ON)
endif()
set(ENABLE_OPENGL OFF)
set(SUYU_NO_JIT ON) # only drops dynarmic, which video_core links but never uses
function(create_target_directory_groups)
endfunction()

# Flags drippu builds with (drippu src/CMakeLists.txt), minus -fno-rtti: the
# runtime uses RTTI and subclasses drippu types (EmuWindow, ...).
if (MSVC) # clang-cl (Windows): the same code generation options through its GCC-style driver
    add_compile_options(/clang:-fwrapv -mcx16 -Wno-array-bounds)
else()
    add_compile_options(-fwrapv -mcx16 -Wno-array-bounds -Wno-stringop-overread -Wno-stringop-overflow)
endif()
add_compile_definitions(BUILD_ID="amd64" NDEBUG _FILE_OFFSET_BITS=64)
if (WIN32) # what drippu's own build defines on Windows (its src/CMakeLists.txt and CMakeLists.txt)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE _SCL_SECURE_NO_WARNINGS NOMINMAX
                            WIN32_LEAN_AND_MEAN UNICODE _UNICODE _WIN32_WINNT=0x0A00 WINVER=0x0A00)
endif()
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/shim)
include_directories(${DRIPPU_SRC})
add_compile_options(-Wno-attributes)

# For code outside this directory that includes drippu headers (the runtime):
# the shims first, then drippu's tree.
add_library(gpu_includes INTERFACE)
target_include_directories(gpu_includes SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/shim ${DRIPPU_SRC})

# video_core's translation units take 1-2 GB each to compile; keep them to 8
# at a time even when the runtime is built with -j28.
set_property(GLOBAL APPEND PROPERTY JOB_POOLS gpu_compile=8)
set(CMAKE_JOB_POOL_COMPILE gpu_compile)

# --- Header-only dependencies -----------------------------------------------

add_library(gpu_boost INTERFACE)
file(GLOB BOOST_INCLUDES ${TP}/boost/libs/*/include ${TP}/boost/libs/numeric/*/include)
target_include_directories(gpu_boost SYSTEM INTERFACE ${BOOST_INCLUDES})
add_library(Boost::headers ALIAS gpu_boost)

add_library(gpu_vulkan_headers INTERFACE)
target_include_directories(gpu_vulkan_headers SYSTEM INTERFACE ${DRIPPU_EXT}/Vulkan-Headers/include)
add_library(Vulkan::Headers ALIAS gpu_vulkan_headers)

add_library(gpu_vulkan_utility INTERFACE)
target_include_directories(gpu_vulkan_utility SYSTEM INTERFACE ${DRIPPU_EXT}/Vulkan-Utility-Libraries/include)
add_library(Vulkan::UtilityHeaders ALIAS gpu_vulkan_utility)

add_library(gpu_vma INTERFACE)
target_include_directories(gpu_vma SYSTEM INTERFACE ${DRIPPU_EXT}/VulkanMemoryAllocator/include)
add_library(GPUOpen::VulkanMemoryAllocator ALIAS gpu_vma)

add_library(gpu_xbyak INTERFACE)
target_include_directories(gpu_xbyak SYSTEM INTERFACE ${DRIPPU_EXT}/xbyak)
add_library(xbyak::xbyak ALIAS gpu_xbyak)

add_library(gpu_unordered_dense INTERFACE)
target_include_directories(gpu_unordered_dense SYSTEM INTERFACE ${DRIPPU_EXT}/unordered_dense/include)
add_library(unordered_dense::unordered_dense ALIAS gpu_unordered_dense)

# --- Compiled dependencies --------------------------------------------------

set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${TP}/fmt fmt EXCLUDE_FROM_ALL)
set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(SIRIT_USE_SYSTEM_SPIRV_HEADERS OFF CACHE BOOL "" FORCE)
add_subdirectory(${DRIPPU_EXT}/sirit sirit EXCLUDE_FROM_ALL)
if (NOT TARGET sirit::sirit)
    add_library(sirit::sirit ALIAS sirit)
endif()

add_library(stb STATIC ${DRIPPU_EXT}/stb/stb_dxt.cpp)
target_include_directories(stb PUBLIC ${DRIPPU_EXT}/stb)
add_library(stb::headers ALIAS stb)

add_library(bc_decoder STATIC ${DRIPPU_EXT}/bc_decoder/bc_decoder.cpp)
target_include_directories(bc_decoder PUBLIC ${DRIPPU_EXT}/bc_decoder)

# glslangValidator compiles the host shaders. drippu's rules run it at
# configure time, so it is built here, once, before anything else.
set(GLSLANG_HOST_DIR ${CMAKE_BINARY_DIR}/glslang-host)
set(GLSLANGVALIDATOR ${GLSLANG_HOST_DIR}/StandAlone/glslang)
if (NOT EXISTS ${GLSLANGVALIDATOR})
    message(STATUS "Building glslang (host tool, once)")
    # Cross builds (Windows from WSL): the tool runs on the build machine, so it is built
    # with the build machine's compilers, not the target's.
    set(GLSLANG_HOST_COMPILERS)
    if (CMAKE_CROSSCOMPILING)
        set(GLSLANG_HOST_COMPILERS -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++)
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env --unset=CC --unset=CXX
                ${CMAKE_COMMAND} -S ${TP}/glslang -B ${GLSLANG_HOST_DIR} -G Ninja ${GLSLANG_HOST_COMPILERS}
                -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=OFF -DENABLE_HLSL=OFF -DGLSLANG_TESTS=OFF
                -DGLSLANG_ENABLE_INSTALL=OFF -DBUILD_EXTERNAL=OFF
        RESULT_VARIABLE GLSLANG_RESULT OUTPUT_QUIET)
    if (GLSLANG_RESULT EQUAL 0)
        execute_process(COMMAND ${CMAKE_COMMAND} --build ${GLSLANG_HOST_DIR} --target glslang-standalone
                        RESULT_VARIABLE GLSLANG_RESULT OUTPUT_QUIET)
    endif()
    if (NOT GLSLANG_RESULT EQUAL 0)
        message(FATAL_ERROR "glslang host build failed")
    endif()
endif()

# FFmpeg (NVDEC/VIC video decoding), configured the way drippu does it
# (externals/ffmpeg/CMakeLists.txt): decoders h264/vp8/vp9 only. No nasm on
# this machine, hence --disable-x86asm.
set(FFMPEG_SRC ${DRIPPU_EXT}/ffmpeg/ffmpeg)
set(FFMPEG_BUILD ${CMAKE_BINARY_DIR}/ffmpeg)
set(FFMPEG_COMPONENTS avfilter swscale avcodec avutil) # link order
set(FFmpeg_LIBRARIES)
foreach (c ${FFMPEG_COMPONENTS})
    list(APPEND FFmpeg_LIBRARIES ${FFMPEG_BUILD}/lib${c}/lib${c}.a)
endforeach()
file(MAKE_DIRECTORY ${FFMPEG_BUILD})
# Windows from WSL: FFmpeg's MSVC toolchain with clang-cl, lld-link and llvm-lib from the
# cross toolchain (tools/cross/windows-clang-cl.cmake), the build machine's compiler for its build tools.
set(FFMPEG_CROSS)
if (CMAKE_CROSSCOMPILING AND WIN32)
    set(FFMPEG_CROSS --target-os=win64 --arch=x86_64 --toolchain=msvc --enable-cross-compile --host-cc=gcc
        --ld=${WINCROSS_LLVM}/lld-link --ar=${WINCROSS_LLVM}/llvm-ar --nm=${WINCROSS_LLVM}/llvm-nm
        "--extra-cflags=${WINCROSS_CFLAGS}" "--extra-ldflags=${WINCROSS_LDFLAGS}")
endif()
add_custom_command(
    OUTPUT ${FFMPEG_BUILD}/Makefile
    COMMAND ${FFMPEG_SRC}/configure ${FFMPEG_CROSS}
            --disable-avdevice --disable-avformat --disable-doc --disable-everything
            --disable-ffmpeg --disable-ffprobe --disable-network --disable-swresample
            --disable-autodetect --disable-x86asm
            --enable-decoder=h264 --enable-decoder=vp8 --enable-decoder=vp9
            --enable-filter=yadif,scale --enable-pic
            --cc=${CMAKE_C_COMPILER} --cxx=${CMAKE_CXX_COMPILER}
    WORKING_DIRECTORY ${FFMPEG_BUILD})
add_custom_command(
    OUTPUT ${FFmpeg_LIBRARIES}
    COMMAND make -j8
    DEPENDS ${FFMPEG_BUILD}/Makefile
    WORKING_DIRECTORY ${FFMPEG_BUILD})
add_custom_target(ffmpeg-build DEPENDS ${FFmpeg_LIBRARIES})
set(FFmpeg_INCLUDE_DIR ${FFMPEG_SRC} ${FFMPEG_BUILD})

# --- common (subset) --------------------------------------------------------

set(COMMON_SOURCES
    address_space.cpp assert.cpp cityhash.cpp cpu_features.cpp dynamic_library.cpp error.cpp
    fs/file.cpp fs/fs.cpp fs/fs_util.cpp fs/path_util.cpp hex_util.cpp host_memory.cpp logging.cpp
    memory_detect.cpp multi_level_page_table.cpp page_table.cpp param_package.cpp settings.cpp
    settings_common.cpp settings_input.cpp steady_clock.cpp stream.cpp string_util.cpp thread.cpp time_zone.cpp uuid.cpp
    virtual_buffer.cpp x64/rdtsc.cpp)
if (WIN32)
    list(APPEND COMMON_SOURCES windows/timer_resolution.cpp)
endif()
list(TRANSFORM COMMON_SOURCES PREPEND ${DRIPPU_SRC}/common/)
# scm_rev.cpp from drippu's template; the "revision" is the drippu submodule's.
execute_process(COMMAND git -C ${DRIPPU} rev-parse --short HEAD OUTPUT_VARIABLE GIT_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
set(GIT_REFSPEC main)
set(GIT_DESC ${GIT_COMMIT})
set(REPO_NAME "drippu (Brothership Recomp)")
set(BUILD_FULLNAME "drippu ${GIT_COMMIT}")
set(BUILD_VERSION ${GIT_COMMIT})
set(BUILD_ID amd64)
set(CXX_COMPILER "GCC ${CMAKE_CXX_COMPILER_VERSION}")
set(IS_DEV_BUILD false)
set(IS_NIGHTLY_BUILD false)
configure_file(${DRIPPU_SRC}/common/scm_rev.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/scm_rev.cpp @ONLY)
add_library(common STATIC ${COMMON_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/scm_rev.cpp)
target_link_libraries(common PUBLIC Boost::headers fmt::fmt stb::headers unordered_dense::unordered_dense
                                    xbyak::xbyak Threads::Threads ${CMAKE_DL_LIBS})
if (WIN32) # ntdll: the timer functions in windows/timer_resolution.cpp; bcrypt: Boost's random generator
    target_link_libraries(common PUBLIC ntdll bcrypt)
endif()

# --- core (shims plus the few real core/ sources video_core needs) ----------

add_library(core STATIC
    ${DRIPPU_SRC}/core/frontend/emu_window.cpp
    ${DRIPPU_SRC}/core/frontend/framebuffer_layout.cpp)
target_link_libraries(core PUBLIC common)

# --- shader_recompiler, gpu_logging: drippu's CMakeLists as they are --------

add_subdirectory(${DRIPPU_SRC}/shader_recompiler shader_recompiler)
add_subdirectory(${DRIPPU_SRC}/video_core/gpu_logging gpu_logging)

# --- Host shaders (drippu video_core/host_shaders/CMakeLists.txt) -----------

function(drippu_list cmake_file block_regex out)
    file(READ ${cmake_file} text)
    string(REGEX MATCH "${block_regex}([^)]*)\\)" block "${text}")
    string(REGEX MATCHALL "[A-Za-z0-9_./]+\\.(cpp|h|inc|vert|frag|comp|glsl)" items "${CMAKE_MATCH_1}")
    set(${out} ${items} PARENT_SCOPE)
endfunction()

set(HS_DIR ${DRIPPU_SRC}/video_core/host_shaders)
drippu_list(${HS_DIR}/CMakeLists.txt "set\\(SHADER_FILES" HS_FILES)
list(TRANSFORM HS_FILES REPLACE "^.*/" "") # ${CMAKE_CURRENT_SOURCE_DIR}/name -> name
set(FIDELITYFX_INCLUDE_DIR ${DRIPPU_EXT}/FidelityFX-FSR/ffx-fsr)
set(HS_OUT ${CMAKE_CURRENT_BINARY_DIR}/host_shaders/include)
set(HS_HEADERS)
foreach (f ${HS_FILES})
    string(REPLACE "." "_" name ${f})
    if (NOT f MATCHES "^vulkan")
        set(h ${HS_OUT}/video_core/host_shaders/${name}.h)
        add_custom_command(OUTPUT ${h}
            COMMAND ${CMAKE_COMMAND} -P ${HS_DIR}/StringShaderHeader.cmake ${HS_DIR}/${f} ${h} ${HS_DIR}/source_shader.h.in
            MAIN_DEPENDENCY ${HS_DIR}/${f} DEPENDS ${HS_DIR}/source_shader.h.in)
        list(APPEND HS_HEADERS ${h})
    endif()
    if (NOT f MATCHES "^opengl")
        string(TOUPPER ${name}_SPV var)
        set(h ${HS_OUT}/video_core/host_shaders/${name}_spv.h)
        add_custom_command(OUTPUT ${h}
            COMMAND ${GLSLANGVALIDATOR} -V --quiet -I${FIDELITYFX_INCLUDE_DIR} -DUseUniformBlock=1
                    --variable-name ${var} -o ${h} ${HS_DIR}/${f} --target-env spirv1.3
            MAIN_DEPENDENCY ${HS_DIR}/${f})
        list(APPEND HS_HEADERS ${h})
    endif()
endforeach()
foreach (f ffx_a.h ffx_fsr1.h)
    string(REPLACE "." "_" name ${f})
    set(h ${HS_OUT}/video_core/host_shaders/${name}.h)
    add_custom_command(OUTPUT ${h}
        COMMAND ${CMAKE_COMMAND} -P ${HS_DIR}/StringShaderHeader.cmake ${FIDELITYFX_INCLUDE_DIR}/${f} ${h} ${HS_DIR}/source_shader.h.in
        MAIN_DEPENDENCY ${FIDELITYFX_INCLUDE_DIR}/${f})
    list(APPEND HS_HEADERS ${h})
endforeach()
add_custom_target(host_shaders DEPENDS ${HS_HEADERS})

# --- video_core -------------------------------------------------------------

drippu_list(${DRIPPU_SRC}/video_core/CMakeLists.txt "add_library\\(video_core STATIC" VC_FILES)
list(FILTER VC_FILES INCLUDE REGEX "\\.cpp$")
list(TRANSFORM VC_FILES PREPEND ${DRIPPU_SRC}/video_core/)
add_library(video_core STATIC ${VC_FILES})
if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # clang-cl reports _MSC_VER but rejects [[msvc::forceinline]] on lambdas, which this file uses when NDEBUG is set.
    set_source_files_properties(${DRIPPU_SRC}/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
                                PROPERTIES COMPILE_OPTIONS -UNDEBUG)
    # The Windows list of FFmpeg hardware decoders names D3D12VA, which the FFmpeg submodule predates (the
    # NVDEC path only; not used by Brothership, whose movies are Bink): tried as D3D11VA a second time.
    set_source_files_properties(${DRIPPU_SRC}/video_core/host1x/ffmpeg.cpp
                                PROPERTIES COMPILE_DEFINITIONS AV_HWDEVICE_TYPE_D3D12VA=AV_HWDEVICE_TYPE_D3D11VA)
endif()
add_dependencies(video_core host_shaders ffmpeg-build)
target_include_directories(video_core PRIVATE ${HS_OUT})
target_include_directories(video_core SYSTEM PUBLIC ${FFmpeg_INCLUDE_DIR})
target_link_libraries(video_core PUBLIC common core shader_recompiler stb bc_decoder gpu_logging
                                        Vulkan::Headers Vulkan::UtilityHeaders GPUOpen::VulkanMemoryAllocator
                                        xbyak::xbyak ${FFmpeg_LIBRARIES})
target_link_libraries(video_core PRIVATE sirit::sirit)
target_link_libraries(video_core INTERFACE gpu_includes)
target_compile_options(video_core PRIVATE -Wno-sign-conversion)

# --- audio_core ---------------------------------------------------------------
# drippu's audio renderer (ADSP, command lists, voices, effects) for audren:u.
# audio_core.cpp (which picks host sinks) is replaced by runtime/src/audio.cpp.

set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(OPUS_INSTALL_PKG_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
set(OPUS_INSTALL_CMAKE_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
add_subdirectory(${DRIPPU_EXT}/opus opus EXCLUDE_FROM_ALL)
if (MSVC AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
    # opus's SSE4.1 files rely on MSVC allowing those instructions anywhere; clang-cl wants them enabled
    # (any CPU Windows 11 runs on has SSE4.2).
    target_compile_options(opus PRIVATE -msse4.1)
endif()

drippu_list(${DRIPPU_SRC}/audio_core/CMakeLists.txt "add_library\\(audio_core STATIC" AC_FILES)
list(FILTER AC_FILES INCLUDE REGEX "\\.cpp$")
list(REMOVE_ITEM AC_FILES audio_core.cpp)
list(TRANSFORM AC_FILES PREPEND ${DRIPPU_SRC}/audio_core/)
add_library(audio_core STATIC ${AC_FILES})
target_link_libraries(audio_core PUBLIC common core Opus::opus)
target_link_libraries(audio_core INTERFACE gpu_includes)
target_compile_options(audio_core PRIVATE -Wno-sign-conversion)
