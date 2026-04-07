cmake_minimum_required(VERSION 3.23)

project(linalg)

# Options
option(LINALG_BUILD_TEST "Build/run the tests" OFF)
option(LINALG_BUILD_EXAMPLES "Build the examples" OFF)

# Directories
if (LINALG_BUILD_TEST)
    add_subdirectory(test)
endif()
if (LINALG_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
add_subdirectory(include)

# Compile Options
add_compile_options(
    -Wall
    -Weffc++
    -Wextra
    -Wconversion
    -Wsign-conversion
    -Werror
)