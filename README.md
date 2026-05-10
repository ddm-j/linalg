# linalg

A bare bones, header-only, linear algebra library written in C++ (without the use of AI)

## Description

`linalg` is my first ever "real" C++ project. The purpose of the project is to demonstrate my understanding of memory management, operator overloads, and some basics of performance optimization.
The library implements the following:
- `Matrix` class
- `MatrixView` and `StrideIterator` that comply with `std::ranges` concepts for iterators
- Partial LU and Cholesky Decomposition
- Linear Solvers based on LU/Cholesky decomp

This is a complete project with a CMake build system, GoogleTest testing, and GoogleBench benchmarking (against Eigen).

## Performance

Performance is not the focus of this project, but here I show how `linalg` performs against the highly optimized library `Eigen`. These results are really only applicable to matrices that fit in the L1 cache (where I focused my performance work)

### Matrix Multiplication

I started with a naive implementation of matrix multiplication. After benchmarking against Eigen, I decided to implement a blocked algorithm to improve cache locality. I focused on the L1 cache alone (for simplicity)

```
------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_Linalg_Matrix_Mul/2         10.8 ns         10.7 ns     65216701
BM_Linalg_Matrix_Mul/4         20.7 ns         20.7 ns     33916454
BM_Linalg_Matrix_Mul/8         85.2 ns         85.1 ns      8228309
BM_Linalg_Matrix_Mul/16         489 ns          488 ns      1416422
BM_Linalg_Matrix_Mul/32        3103 ns         3097 ns       206410
BM_Linalg_Matrix_Mul/64       21427 ns        21399 ns        32784
BM_Linalg_Matrix_Mul/128     181350 ns       181112 ns         3846
BM_Linalg_Matrix_Mul/256    1452932 ns      1450507 ns          481
BM_Linalg_Matrix_Mul/512   12128805 ns     12105841 ns           59
BM_Linalg_Matrix_Mul_BigO       0.09 N^3        0.09 N^3  
BM_Linalg_Matrix_Mul_RMS          1 %             1 %    
BM_Eigen_Matrix_Mul/2          9.12 ns         9.11 ns     76104897
BM_Eigen_Matrix_Mul/4          14.5 ns         14.5 ns     45786874
BM_Eigen_Matrix_Mul/8           128 ns          128 ns      5483972
BM_Eigen_Matrix_Mul/16          388 ns          387 ns      1804962
BM_Eigen_Matrix_Mul/32         4457 ns         4451 ns       156029
BM_Eigen_Matrix_Mul/64        23337 ns        23310 ns        30036
BM_Eigen_Matrix_Mul/128      159332 ns       159117 ns         4401
BM_Eigen_Matrix_Mul/256     1170141 ns      1168209 ns          599
BM_Eigen_Matrix_Mul/512     9376939 ns      9360713 ns           75
BM_Eigen_Matrix_Mul_BigO       0.07 N^3        0.07 N^3  
BM_Eigen_Matrix_Mul_RMS           0 %             0 %    
```

As shown above for double filled `Matrix`, Eigen's `MatrixXd` is only about 30% faster than `linalg`.

### Linear Solvers

- The LU decomposition based solver is about the same speed as Eigen's implementation.
- The cholesky implementation is faster than Eigen's.
(for matrices up to 512x512)

More detailed results are available by building the benchmarking target.

## Getting Started

### Dependencies

Dependencies are controlled by the build target, and are pulled/managed solely by CMake.
* **Test Target**: GoogleTest
* **Benchmarking Target**: GoogleBench, Eigen

### Example Use

See `examples.cpp` or the examples build target for more:

```c++
#include <linalg/Matrix.h>
#include <linalg/solvers.h>

// Matrix Multiplication
Matrix<int> A {{1,2},{3,4}};
Matrix<int> B {{1,2,3,4},{5,6,7,8}};
auto C { A * B };
    
// Linear Solver (LU)
Matrix<double> D {
    { 0, 1, 2 },
    { 1, 3, 1 },
    { 2, 1, 4 }
};
Matrix<double> x(3, 1, {1, 2, 3});
auto x_sol { linalg::solvers::solve_lu(D, D*x) };
```


## Acknowledgments

learncpp.com