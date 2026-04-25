#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include <utility>
#include <algorithm>
#include <climits>
#include <exception>
#include <linalg/Matrix.h>

using linalg::Matrix;

namespace linalg::decomposition {

template <typename T>
std::tuple<Matrix<typename Matrix<T>::size_type>, Matrix<T>, Matrix<T>> lu(const Matrix<T>& A, int fac = 100)
{
    using size_type = Matrix<T>::size_type;
    size_type rows { A.rows() };
    size_type cols { A.cols() };
    T eps { fac * std::numeric_limits<T>::epsilon() };

    Matrix<T> L { Matrix<T>::eye(rows, rows) };
    Matrix<T> U { A };
    Matrix<size_type> P(rows, 1);
    size_type n { 0 };
    std::generate(P.begin(), P.end(), [&n](){ return n++; });

    // Loop Down the Diagonal of U
    T f {};
    T rMax {};
    size_type iMax {};

    for (size_type d {0}; d < std::min(rows, cols); ++d) 
    {
        // Partial Pivoting
        rMax = std::abs(U[d, d]);
        iMax = d;
        for (size_type i {d+1}; i < rows; ++i)
        {
            if (std::abs(U[i, d]) > rMax)
            {
                rMax = std::abs(U[i, d]);
                iMax = i;
            }
        }
        if (iMax != d)
        {
            auto ur0 { U.rowit(d) };            
            auto ur1 { U.rowit(iMax) };
            auto lr0 { L.rowit(d) };            
            auto lr1 { L.rowit(iMax) };
            std::swap_ranges(ur0.begin(), ur0.end(), ur1.begin());
            std::swap_ranges(lr0.begin(), lr0.begin() + d , lr1.begin());
            std::swap(P[d], P[iMax]);
        }

        // Singular Matrix
        if (abs(U[d, d]) < eps)
            throw std::runtime_error("Cannot LU factor a singular matrix.");

        // Elimination
        for (size_type i {d+1}; i < rows; ++i)
        {
            f = U[i, d] / U[d, d];
            L[i, d] = f;
            for (size_type j {d}; j < cols; ++j)
            {
                U[i, j] -= f*U[d, j];
            }
            std::cout << "\n";
        }
    }
    return std::tuple { P, L, U };
}

}

#endif // DECOMPOSITION_H