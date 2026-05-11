# linalg

A bare bones, header-only, linear algebra library written in C++ - **by human hand**

## Description

`linalg` is my first ever "real" C++ project. The purpose of the project is to demonstrate my understanding of memory management, operator overloads, and some basics of performance optimization.
The library implements the following:
- `Matrix` class
- `MatrixView` and `StrideIterator` compliant with `std::bidirectional_iterator` concept
- Partial LU and Cholesky Decomposition
- Linear Solvers based on LU/Cholesky decomp

This is a complete project with a CMake build system, GoogleTest testing, and GBench benchmarking (against Eigen).

## Architecture Notes

The layering of this repo is as follows: **errors** → **views** → **Matrix** → **decomposition/solvers** 

#### Errors
A lightweight header deriving `linalg` error types from `std::invalid_argument` (design choice)

#### Views
Mutable or non-mutable views into the `Matrix` class elements using a configurable, strided iterator. Primarily used for `Matrix` transpose (non allocating), row/column views, and diagonal view.

#### Matrix
The `Matrix` class is the core of the library. Elements are stored in **row-major** order, using `std::unique_ptr<T[]>` for allocation, and full "rule of 5" RAII. 

#### Decomposition / Solvers
I've only implemented PLU and Cholesky decomposition (and linear solvers).


## Performance

Performance is not the focus of this project, but here I show how `linalg` performs against the highly optimized library `Eigen`. These results are really only applicable to matrices that fit in the L1 cache (where I focused my performance work)

### Matrix Multiplication

I started with a naive implementation of matrix multiplication. After benchmarking against Eigen, I decided to implement a blocked algorithm to improve cache locality. I focused on the L1 cache alone (for simplicity)

```
AMD 9800 X3D
CPU Caches:
  L1 Data 48 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1024 KiB (x8)
  L3 Unified 98304 KiB (x1)
Load Average: 1.12, 0.57, 0.43
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_Linalg_Matrix_Mul/2         13.8 ns         13.8 ns     51041263
BM_Linalg_Matrix_Mul/4         31.1 ns         31.1 ns     22536592
BM_Linalg_Matrix_Mul/8          104 ns          104 ns      6713767
BM_Linalg_Matrix_Mul/16         639 ns          638 ns      1094552
BM_Linalg_Matrix_Mul/32        4252 ns         4246 ns       164804
BM_Linalg_Matrix_Mul/64       31210 ns        31172 ns        22455
BM_Linalg_Matrix_Mul/128     278590 ns       278253 ns         2514
BM_Linalg_Matrix_Mul/256    2301293 ns      2297515 ns          305
BM_Linalg_Matrix_Mul/512   19248147 ns     19215962 ns           36
BM_Linalg_Matrix_Mul_BigO       0.14 N^3        0.14 N^3  
BM_Linalg_Matrix_Mul_RMS          1 %             1 %    
BM_Eigen_Matrix_Mul/2          8.73 ns         8.72 ns     73576777
BM_Eigen_Matrix_Mul/4          10.5 ns         10.5 ns     66475332
BM_Eigen_Matrix_Mul/8          42.5 ns         42.4 ns     16490207
BM_Eigen_Matrix_Mul/16          110 ns          110 ns      6374234
BM_Eigen_Matrix_Mul/32          619 ns          618 ns      1123776
BM_Eigen_Matrix_Mul/64         4165 ns         4160 ns       168249
BM_Eigen_Matrix_Mul/128       29676 ns        29640 ns        23595
BM_Eigen_Matrix_Mul/256      229640 ns       229245 ns         3054
BM_Eigen_Matrix_Mul/512     2149427 ns      2145472 ns          329
BM_Eigen_Matrix_Mul_BigO       0.02 N^3        0.02 N^3  
BM_Eigen_Matrix_Mul_RMS           5 %             5 %    
```

As shown above for double filled `Matrix`, Eigen's `MatrixXd` is about 7X faster than `linalg` (`-march=native`).
By compiling with `LINALG_BLOCKED_MATMUL` off, the naive algorithm is 70X slower than Eigen.

### Linear Solvers

- The LU decomposition based solver is about the same speed as Eigen's implementation.
- The cholesky implementation is faster than Eigen's.
(for matrices up to 512x512)

More detailed results are available by building the benchmarking target.

## Getting Started

### Dependencies

Dependencies are controlled by the build target, and are pulled/managed solely by CMake.
* **Test Target**: GoogleTest
* **Benchmarking Target**: GBench, Eigen

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