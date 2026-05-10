#include <iostream>
#include <algorithm>
#include <format>
#include <linalg/Matrix.h>
#include <linalg/solvers.h>
#include <linalg/decomposition.h>

int main()
{
    using linalg::Matrix;

    // Matrix Class
    // // Nested Initializer List
    Matrix<int> A {{1,2},{3,4}};
    // // Initializer List with Shape
    Matrix<double> B(1, 3, {1, 2, 3});
    // // Arbitrary Shape (zero initialized)
    Matrix<double> C(2, 2);

    // Multiplication
    Matrix<int> D {{1,2,3,4},{5,6,7,8}};
    Matrix<int> F { A * D };
    
    // Matrix Output (formatted)
    std::cout << "A * D = \n" << F << "\n";

    // PLU Decomposition
    Matrix<double> G {
        { 0, 1, 2 },
        { 1, 3, 1 },
        { 2, 1, 4 }
    };
    auto [P, L, U] { linalg::decomposition::lu(G) };

    // Cholesky Decomposition
    Matrix<double> H {
        {6, 3, 4, 8},
        {3, 6, 5, 1},
        {4, 5,10, 7},
        {8, 1, 7,25}
    };
    auto I { linalg::decomposition::cholesky(H) };

    // Linear Solver (LU Decomp)
    Matrix<double> J {
        { 0, 1, 2 },
        { 1, 3, 1 },
        { 2, 1, 4 }
    };
    Matrix<double> xJ(3, 1, {1, 2, 3});
    auto xJ_sol { linalg::solvers::solve_lu(J, J*xJ) };

    // Linear Solver (Cholesky Decomop)
    Matrix<double> K {
        {6, 3, 4, 8},
        {3, 6, 5, 1},
        {4, 5,10, 7},
        {8, 1, 7,25}
    };
    Matrix<double> xK(4, 1, {1, 2, 3, 4});
    auto xK_sol { linalg::solvers::solve_cholesky(K, K*xK) };

    return 0;
}