# A Windows executable that lists the Vulkan devices: checks the cross toolchain (tools/cross) end to end.
#   cmake -S tools/cross/vkprobe -B build/win/vkprobe -G Ninja -DCMAKE_TOOLCHAIN_FILE=$PWD/tools/cross/windows-clang-cl.cmake
#   ninja -C build/win/vkprobe   ->  build/win/vkprobe/vkprobe.exe, run on Windows
cmake_minimum_required(VERSION 3.20)
project(vkprobe CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded) # static C/C++ runtime: no Visual C++ redistributable needed
add_executable(vkprobe main.cpp)
target_include_directories(vkprobe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/drippu/externals/Vulkan-Headers/include)
