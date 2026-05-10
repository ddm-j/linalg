#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include <utility>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cerrno>
#include <format>
#include <linalg/errors.h>
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
        if (std::abs(U[d, d]) < eps)
            throw linalg::Singular(std::format("lu(): Matrix is singular, U[{},{}]=0",d,d));

        // Elimination
        for (size_type i {d+1}; i < rows; ++i)
        {
            f = U[i, d] / U[d, d];
            L[i, d] = f;
            for (size_type j {d}; j < cols; ++j)
            {
                U[i, j] -= f*U[d, j];
            }
        }
    }
    return std::tuple { P, L, U };
}

template <typename T>
Matrix<T> cholesky(const Matrix<T>& A, int fac = 10)
{
    using size_type = Matrix<T>::size_type;
    size_type rows { A.rows() };
    size_type cols { A.cols() };
    T eps { fac * std::numeric_limits<T>::epsilon() };

    // Shape Check
    if (rows != cols)
        throw linalg::ShapeError("cholesky(): Matrix is not square.");
    if (A != A.transpose())
        throw linalg::NonSymmetric("cholesky(): Matrix is not symmetric.");
    

    Matrix<T> L(rows, rows);

    T tmp_sum {};
    for (size_type k {0}; k < rows; ++k)
    {
        // On Diagonal
        tmp_sum = T{};
        errno = 0;
        for (size_type j {0}; j < k; ++j)
        {
            tmp_sum += std::pow(L[k,j], 2);
        }
        L[k,k] = std::sqrt(A[k,k] - tmp_sum);
        if (errno == EDOM) // SQRT of negative check
            throw linalg::Indefinite(std::format("cholesky(): Matrix is not positive definite, k={} diagonal.", k));
        if (std::abs(L[k,k]) < eps) // Divide by zero check
            throw linalg::Singular(std::format("cholesky(): Matrix is singular. k={} diagonal.", k));

        // Off Diagonal
        for (size_type i {k + 1}; i < rows; ++i)
        {
            tmp_sum = T{};
            for (size_type j {0}; j < k; ++j)
            {
                tmp_sum += L[i,j] * L[k,j];
            }
            L[i,k] = (1.0 / L[k,k]) * (A[i,k] - tmp_sum);
        }
    }

    return L;
}
}

#endif // DECOMPOSITION_H