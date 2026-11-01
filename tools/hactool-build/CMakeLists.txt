# hactool (third_party/hactool, SciresM, ISC; with its bundled mbedtls, Apache-2.0) for the game builder's unpack stage
# (the player's .nsp files and keys -> exefs/ + romfs/). Its Makefile's sources and defines;
# Windows (clang-cl, tools/cross): Microsoft's C library has no getopt_long - ours in compat/. tools/make_toolchain.sh
# builds it into the release's tools/.
cmake_minimum_required(VERSION 3.20)
project(hactool C)

set(H "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/hactool")
file(GLOB MBEDTLS "${H}/mbedtls/library/*.c")
set(SOURCES save sha aes extkeys rsa npdm bktr kip packages pki pfs0 hfs0 nca0_romfs romfs utils nax0 nso lz4 nca xci
            main filepath ConvertUTF cJSON)
list(TRANSFORM SOURCES PREPEND "${H}/")
list(TRANSFORM SOURCES APPEND ".c")
add_executable(hactool ${SOURCES} ${MBEDTLS})
target_include_directories(hactool PRIVATE "${H}/mbedtls/include" "${H}")
target_compile_definitions(hactool PRIVATE _BSD_SOURCE _POSIX_SOURCE _POSIX_C_SOURCE=200112L _DEFAULT_SOURCE
                                           _FILE_OFFSET_BITS=64)
if (MSVC)
    target_sources(hactool PRIVATE compat/getopt.c)
    target_include_directories(hactool PRIVATE compat)
    target_compile_definitions(hactool PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(hactool PRIVATE /w)
    target_link_libraries(hactool PRIVATE advapi32)
else()
    target_compile_options(hactool PRIVATE -w -std=gnu11)
endif()
