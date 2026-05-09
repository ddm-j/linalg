add_executable(benchmark_linalg)

target_sources(benchmark_linalg
    PRIVATE
    benchmarks.cpp
)

include(FetchContent)

# Eigen
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.zip 
)
FetchContent_MakeAvailable(eigen)

# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googlebench
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.zip
)
FetchContent_MakeAvailable(googlebench)

target_link_libraries(benchmark_linalg
    PRIVATE
        benchmark::benchmark
        Eigen3::Eigen
        linalg
)