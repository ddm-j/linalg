#include <iostream>
#include <algorithm>
#include <format>
#include <linalg/Matrix.h>
#include <linalg/stride_view.h>
#include <linalg/solvers.h>
#include <linalg/decomposition.h>

int main()
{
    Matrix<double> A {
        {6, 3, 4, 8, 10, -1, 2},
        {3, 6, 5, 1, 12, -5, 3},
        {4, 5,10, 7, 13, -9, -1},
        {8, 1, 7,25, 2, -2, -4}
    };

    Matrix<double> B { A.transpose() };

    auto C { A*B };

    return 0;
}