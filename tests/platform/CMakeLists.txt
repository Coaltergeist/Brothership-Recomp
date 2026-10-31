# The platform layer's tests, built on their own: the guest-memory test (memory_test.cpp) and the crash report
# (crash_test.cpp). Linux with posix.cpp, Windows (cross toolchain, tools/cross/windows-clang-cl.cmake) with
# windows.cpp. tests/platform/run.sh drives them.
cmake_minimum_required(VERSION 3.20)
project(platform_test CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(RT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/src/platform")
foreach(test memory_test crash_test)
    if (WIN32)
        add_executable(${test} ${test}.cpp "${RT_SRC}/windows.cpp")
        target_link_libraries(${test} PRIVATE winmm)
    else()
        find_package(Threads REQUIRED)
        add_executable(${test} ${test}.cpp "${RT_SRC}/posix.cpp")
        target_link_libraries(${test} PRIVATE Threads::Threads)
    endif()
    target_include_directories(${test} PRIVATE "${RT_SRC}")
endforeach()
