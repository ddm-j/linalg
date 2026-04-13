add_executable(test_linalg)

target_sources(test_linalg
    PRIVATE
    test_matrix.cpp
    test_strideview.cpp
)

if(ENABLE_COVERAGE)
    find_program(LCOV lcov REQUIRED)
    find_program(GENHTML genhtml REQUIRED)

    target_compile_options(test_linalg PRIVATE --coverage -O0)
    target_link_options(test_linalg PRIVATE --coverage)

    add_custom_target(coverage
        COMMAND ${LCOV} --directory . --capture --output-file coverage.info --ignore-errors mismatch
        COMMAND ${LCOV} --remove coverage.info '/usr/*' '*/googletest/*' --output-file coverage_filtered.info --ignore-errors mismatch
        COMMAND ${GENHTML} --demangle-cpp -o coverage coverage_filtered.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS test_linalg
    )
endif()

target_compile_features(test_linalg PRIVATE cxx_std_17)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

target_link_libraries(test_linalg
    PRIVATE
        GTest::gtest_main
        linalg
)

include(GoogleTest)
gtest_discover_tests(test_linalg)