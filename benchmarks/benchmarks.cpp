#include <string>
#include <linalg/Matrix.h>
#include <linalg/solvers.h>
#include <random>
#include <Eigen/Dense>
#include <benchmark/benchmark.h>

// Helpers
linalg::Matrix<double> make_random(int n, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    linalg::Matrix<double> m(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m[i, j] = dist(rng);
    return m;
}

Eigen::MatrixXd to_eigen(const linalg::Matrix<double>& m)
{
  auto n {m.rows()};
  Eigen::MatrixXd M(n, n);
  for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
          M(i, j) = m[i, j];
  return M;
}

// Matrix Benchmarks (multiplication only)
static void BM_Linalg_Matrix_Mul(benchmark::State& state) {
  int n = state.range(0);
  linalg::Matrix<double> A(n, n), B(n, n);
  for (auto _ : state)
  {
    linalg::Matrix<double> C { A * B };
    benchmark::DoNotOptimize(C);
  }
  state.SetComplexityN(n);
}
BENCHMARK(BM_Linalg_Matrix_Mul)->RangeMultiplier(2)->Range(2, 512)->Complexity(benchmark::oNCubed);

static void BM_Eigen_Matrix_Mul(benchmark::State& state) {
  int n = state.range(0);
  Eigen::MatrixXd A(n, n), B(n, n);
  for (auto _ : state)
  {
    Eigen::MatrixXd C { A * B };
    benchmark::DoNotOptimize(C);
  }
  state.SetComplexityN(n);
}
BENCHMARK(BM_Eigen_Matrix_Mul)->RangeMultiplier(2)->Range(2, 512)->Complexity(benchmark::oNCubed);

// PLU Solver Benchmarks
static void BM_Linalg_PLU(benchmark::State& state) {
  int n = state.range(0);
  linalg::Matrix<double> R { make_random(n) };
  auto x_true { linalg::Matrix<double>::ones(n, 1) };
  auto A { R + static_cast<double>(n)*linalg::Matrix<double>::eye(n, n) };
  auto b { A * x_true };
  for (auto _ : state)
  {
    linalg::Matrix<double> x { linalg::solvers::solve_lu(A, b) };
    benchmark::DoNotOptimize(x);
  }
  state.SetComplexityN(n);
}
BENCHMARK(BM_Linalg_PLU)->RangeMultiplier(2)->Range(2, 512)->Complexity(benchmark::oNCubed);

static void BM_Eigen_PLU(benchmark::State& state) {
  int n = state.range(0);
  Eigen::MatrixXd R { to_eigen(make_random(n)) };
  auto x_true { Eigen::MatrixXd::Ones(n, 1) };
  auto A { R + static_cast<double>(n)*Eigen::MatrixXd::Identity(n, n) };
  auto b { A * x_true };
  for (auto _ : state)
  {
    Eigen::MatrixXd x { A.partialPivLu().solve(b) };
    benchmark::DoNotOptimize(x);
  }
  state.SetComplexityN(n);
}
BENCHMARK(BM_Eigen_PLU)->RangeMultiplier(2)->Range(2, 512)->Complexity(benchmark::oNCubed);

// Cholesky Solver Benchmarks
static void BM_Linalg_Cholesky(benchmark::State& state) {
  int n = state.range(0);
  linalg::Matrix<double> M { make_random(n) };
  auto x_true { linalg::Matrix<double>::ones(n, 1) };
  double eps { std::numeric_limits<double>::epsilon() };
  auto A { M.transpose() * M + 10000*eps*linalg::Matrix<double>::eye(n, n) };
  auto b { A * x_true };
  for (auto _ : state)
  {
    linalg::Matrix<double> x { linalg::solvers::solve_cholesky(A, b) };
    benchmark::DoNotOptimize(x);
  }
  state.SetComplexityN(n);
}
BENCHMARK(BM_Linalg_Cholesky)->RangeMultiplier(2)->Range(2, 512)->Complexity(benchmark::oNCubed);

static void BM_Eigen_Cholesky(benchmark::State& state) {
  int n = state.range(0);
  Eigen::MatrixXd M { to_eigen(make_random(n)) };
  auto x_true { Eigen::MatrixXd::Ones(n, 1) };
  double eps { std::numeric_limits<double>::epsilon() };
  auto A { M.transpose() * M + 10000*eps*Eigen::MatrixXd::Identity(n, n) };
  auto b { A * x_true };
  for (auto _ : state)
  {
    Eigen::MatrixXd x { A.llt().solve(b) };
    benchmark::DoNotOptimize(x);
  }
  state.SetComplexityN(n);
}
BENCHMARK(BM_Eigen_Cholesky)->RangeMultiplier(2)->Range(2, 512)->Complexity(benchmark::oNCubed);

BENCHMARK_MAIN();