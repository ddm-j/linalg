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
        {6, 3, 4, 8},
        {3, 6, 5, 1},
        {4, 5,10, 7},
    };

    // linalg::MatrixView<double> view { A }; 

    linalg::MatrixView<double> view(A.begin(), A.cols(), A.rows(), 1, A.cols()); 
    Matrix<double> B { view };
    auto C {A * view};
    std::cout << A << "\n\n";
    std::cout << B << "\n\n";
    std::cout << C << "\n\n";

    // for (Matrix<double>::size_type i {0}; i < A.length(); ++i)
    // {
    //     std::cout << std::format("i={} -- {}, {}\n",i,A[i],view[i]);
    // }

    // auto L { linalg::decomposition::cholesky(A) };
    // auto Lt { L.transpose() };
    // std::cout << L << "\n\n";
    // std::cout << Lt << "\n\n";
    // std::cout << L * Lt << "\n\n";
    return 0;
}