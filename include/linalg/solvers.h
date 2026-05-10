#ifndef SOLVERS_H
#define SOLVERS_H

#include <utility>
#include <algorithm>
#include <exception>
#include <linalg/errors.h>
#include <linalg/Matrix.h>
#include <linalg/decomposition.h>

using linalg::Matrix;

namespace linalg::solvers {

template <typename T>
Matrix<T> forward_sub(const Matrix<T>& L, const Matrix<T>& b)
{
    using size_type = Matrix<T>::size_type;

    size_type rows { b.rows() };

    // Ensure Lower Tri
    if (!is_lowertri(L))
        throw linalg::NonLowerTriangular("forward_sub(): Matrix L is not lower triangular");

    // Solution Vector
    Matrix<T> y(rows, 1);

    y[0] = b[0] / L[0,0];
    T sum {};
    for (size_type i {1}; i < rows; ++i)
    {
       sum = T{}; 
       for (size_type j {0}; j < i; ++j)
       {
            sum += L[i,j] * y[j];
       }
       y[i] = (b[i] - sum) / L[i,i];
    }
    return y;
}

template <typename T>
Matrix<T> backward_sub(const Matrix<T>& U, const Matrix<T>& y)
{
    using size_type = Matrix<T>::size_type;
    size_type rows { y.rows() };

    // Ensure Upper Tri
    if (!is_uppertri(U))
        throw linalg::NonUpperTriangular("backward_sub(): Matrix U is not upper triangular");

    // Solution Vector
    Matrix<T> x (rows, 1);

    size_type i {};
    T sum {};
    for (size_type it { rows }; it > 0; --it)
    {
        sum = T{};
        i = it - 1;
        for (size_type j {i + 1}; j < rows; ++j)
        {
            sum += U[i, j]*x[j];
        }
        x[i] = (y[i] - sum) / U[i, i];
    }
    return x;
}

template <typename T>
Matrix<T> solve_lu(const Matrix<T>& A, const Matrix<T>& b)
{
    using size_type = Matrix<T>::size_type;
    size_type rows = A.rows();
    size_type cols = A.cols();

    // Size Checking
    if (rows != cols)
        throw linalg::ShapeError("solve_lu(): Matrix A must be square.");
    if (cols != b.rows())
        throw linalg::ShapeError("solve_lu(): Matrix b must have same number of rows as A has columns.");
    if (b.cols() > 1)
        throw linalg::ShapeError("solve_lu(): Matrix b cannot have more than one column");

    // LU Decomp
    auto [ P, L, U ] { linalg::decomposition::lu(A) };

    // Permute b
    Matrix<T> Pb(b.rows(), 1);
    std::transform(P.begin(), P.end(), Pb.begin(),
        [&b](const auto& a)
        {
            return b[a]; 
        }
    );

    // Forward Substitution
    Matrix<T> y { forward_sub(L, Pb) };

    // Backward Substitution
    Matrix<T> x { backward_sub(U, y)};

    return x;
}

template <typename T>
Matrix<T> solve_cholesky(const Matrix<T>& A, const Matrix<T>& b)
{
    using size_type = Matrix<T>::size_type;
    size_type rows = A.rows();
    size_type cols = A.cols();

    // Size Checking
    if (rows != cols)
        throw linalg::ShapeError("solve_cholesky(): Matrix A must be square.");
    if (cols != b.rows())
        throw linalg::ShapeError("solve_cholesky(): Matrix b must have same number of rows as A has columns.");
    if (b.cols() > 1)
        throw linalg::ShapeError("solve_cholesky(): Matrix b cannot have more than one column");

    // Decompose
    auto L { linalg::decomposition::cholesky(A) };

    // Forward Substitution
    Matrix<T> y { forward_sub(L, b) };

    // Backward Substitution
    Matrix<T> x { backward_sub(Matrix<T>(L.transpose()), y) };

    return x;
}

}

#endif // SOLVERS_H