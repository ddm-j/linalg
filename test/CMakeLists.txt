add_executable(test_linalg)

target_sources(test_linalg
    PRIVATE
    test_matrix.cpp
)

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