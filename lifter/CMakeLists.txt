cmake_minimum_required(VERSION 3.20)
project(nxrecomp_lifter CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DRIPPU_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/drippu/src" CACHE PATH
    "drippu source tree providing core/recompiler/arm64_to_c.h")

add_executable(nxrecomp src/main.cpp src/image.cpp src/analysis.cpp src/emit.cpp src/link.cpp src/simd.cpp src/fastmem.cpp src/patch.cpp)
target_include_directories(nxrecomp PRIVATE "${DRIPPU_SRC}")
target_compile_options(nxrecomp PRIVATE -Wall -Wextra)
