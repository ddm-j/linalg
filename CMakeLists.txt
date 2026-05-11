cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(linalg)

# Options
option(LINALG_BUILD_TEST "Build/run the tests" OFF)
option(LINALG_BUILD_EXAMPLES "Build the examples" OFF)
option(LINALG_BUILD_BENCH "Build the benchmarks" OFF)
option(LINALG_BLOCKED_MATMUL "Use blocked matrix multiplication algorithm" ON)

# Directories
if (LINALG_BUILD_TEST)
    add_subdirectory(test)
endif()
if (LINALG_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if (LINALG_BUILD_BENCH)
    add_subdirectory(benchmarks)
endif()
add_subdirectory(include)
