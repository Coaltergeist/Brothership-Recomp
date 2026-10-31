# CMake toolchain: Windows x86-64 executables built in WSL (PORT_PLAN Phase 4).
#
#   cmake -S <src> -B build/win/<name> -G Ninja \
#         -DCMAKE_TOOLCHAIN_FILE=$PWD/tools/cross/windows-clang-cl.cmake
#
# Compilers and linker from the LLVM release in tools/llvm (clang-cl, lld-link,
# llvm-lib, llvm-rc); Microsoft's headers and import libraries from the copy of
# Visual Studio 2022 (MSVC 14.44) and the Windows SDK 10.0.26100 in tools/winsdk,
# made usable on a case-sensitive filesystem by tools/cross/winsdk_aliases.py.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 10.0)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(_llvm "${_root}/tools/llvm/bin")
set(_w "${_root}/tools/winsdk")

set(CMAKE_C_COMPILER "${_llvm}/clang-cl")
set(CMAKE_CXX_COMPILER "${_llvm}/clang-cl")
set(CMAKE_C_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_LINKER "${_llvm}/lld-link")
set(CMAKE_AR "${_llvm}/llvm-lib")
set(CMAKE_RC_COMPILER "${_llvm}/llvm-rc")
set(CMAKE_MT "${_llvm}/llvm-mt")

# Headers: /imsvc marks them as system headers (no warnings), as the MSVC environment would. (Paths without
# quotes: the checkout's path must not contain spaces.) WINCROSS_CFLAGS / WINCROSS_LDFLAGS also serve builds that
# do not use CMake (FFmpeg's configure, runtime/gpu).
set(WINCROSS_LLVM "${_llvm}")
set(WINCROSS_CFLAGS "")
foreach (d msvc/include sdk/include/ucrt sdk/include/um sdk/include/shared sdk/include/winrt)
    string(APPEND WINCROSS_CFLAGS " /imsvc ${_w}/${d}")
endforeach()
string(STRIP "${WINCROSS_CFLAGS}" WINCROSS_CFLAGS)
# MSVC-only options some projects add (/MP, /Zc:throwingNew) are ignored by clang-cl; not an error under -Werror.
string(APPEND CMAKE_C_FLAGS_INIT " ${WINCROSS_CFLAGS} -Wno-unused-command-line-argument")
string(APPEND CMAKE_CXX_FLAGS_INIT " ${WINCROSS_CFLAGS} -Wno-unused-command-line-argument")
string(APPEND CMAKE_RC_FLAGS_INIT " -I ${_w}/sdk/include/um -I ${_w}/sdk/include/shared")

# Import libraries and the C/C++ runtime libraries.
set(WINCROSS_LDFLAGS "")
foreach (d msvc/lib/x64 sdk/lib/ucrt/x64 sdk/lib/um/x64)
    string(APPEND WINCROSS_LDFLAGS " /libpath:${_w}/${d}")
endforeach()
string(STRIP "${WINCROSS_LDFLAGS}" WINCROSS_LDFLAGS)
foreach (kind EXE SHARED MODULE)
    string(APPEND CMAKE_${kind}_LINKER_FLAGS_INIT " ${WINCROSS_LDFLAGS}")
endforeach()

# Look for libraries and headers only in the target environment, programs on the build machine.
set(CMAKE_FIND_ROOT_PATH "${_w}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
