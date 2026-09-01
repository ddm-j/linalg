#ifndef LINALG_SOLVERS_H
#define LINALG_SOLVERS_H

#include <utility>
#include <algorithm>
#include <exception>
#include <cmath>
#include <linalg/errors.h>
#include <linalg/Matrix.h>
#include <linalg/decomposition.h>

namespace linalg::solvers {

using linalg::Matrix;

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

template <typename T>
void solve_gauss_jordan(Matrix<T>& A, Matrix<T>& B, const T tol = 0.0)
{
    using std::abs;

    // Numerical Recipes - Third Edition Implementation +
    // Mutates A and B (caller beware)
    using size_type = Matrix<T>::size_type;
    const size_type rows = A.rows();
    const size_type cols = A.cols();

    // Size Checking
    if (rows != cols)
        throw linalg::ShapeError("solve_gauss_jordan(): Matrix A must be square.");
    if (cols != B.rows())
        throw linalg::ShapeError("solve_gauss_jordan(): Matrix b must have same number of rows as A has columns.");
    
    // Setup
    size_type i {}, icol {}, irow {};
    size_type j {}, k {}, l {}, ll {};
    size_type n { A.rows() }, m { B.cols() };
    T big {}, dum {}, pivinv {};
    Matrix<size_type> indxc(1, n), indxr(1, n), ipiv(1, n);
    std::fill(ipiv.begin(), ipiv.end(), 0);

    // Singular Detection Tolerance
    const T A_max { *(std::max_element(A.begin(), A.end(), [](T a, T b){ return std::abs(a) < std::abs(b); })) };
    // Auto Tolerance Calculation
    const T eps { std::numeric_limits<T>::epsilon() };
    const T pivot_tol { (tol < 0.0)
        ? static_cast<T>(n) * eps * A_max
        : tol * A_max };

    // Main Loop
    for (i = 0; i < n; i++)
    {
        big = T{};
        for (j = 0; j < n; j++)
            if (ipiv[j] != 1)
                for (k = 0; k < n; k++) {            
                    if (ipiv[k] == 0) {
                        if (abs(A[j, k]) >= big) {
                            big = abs(A[j, k]);
                            irow = j;
                            icol = k;
                        }
                    }
                }
        ++(ipiv[icol]);
        // Row Swapping
        if (irow != icol) {
            std::swap_ranges(A.rowit(irow).begin(), A.rowit(irow).end(), A.rowit(icol).begin());
            std::swap_ranges(B.rowit(irow).begin(), B.rowit(irow).end(), B.rowit(icol).begin());
        }
        indxr[i] = irow;
        indxc[i] = icol;

        // Detect Singular Matrix (uses tolerance, deviation from NR)
        if (abs(A[icol, icol]) <= pivot_tol) throw linalg::Singular("solve_gauss_jordan(): Matrix is singular");
        // Divide Pivot Row by the pivot element
        pivinv = T{1} / A[icol, icol];
        A[icol, icol] = T{1};
        for (auto& x : A.rowit(icol)) x *= pivinv;
        for (auto& x : B.rowit(icol)) x *= pivinv;

        // Reduce the row
        for (ll = 0; ll < n; ll++) {
            if (ll != icol) {
                dum = A[ll, icol];
                A[ll, icol] = T{};
                for (l = 0; l < n; l++) A[ll, l] -= A[icol, l]*dum;
                for (l = 0; l < m; l++) B[ll, l] -= B[icol, l]*dum;
            }
        }
    }
    // Unscrample Solution
    for (l = n; l > 0; --l) // size_type will wraparound (differs from NR slightly)
    {
        if (indxr[l-1] != indxc[l-1])
        {
            std::swap_ranges(A.colit(indxr[l-1]).begin(), A.colit(indxr[l-1]).end(), A.colit(indxc[l-1]).begin());
        }
    }

}

}

#endif // LINALG_SOLVERS_H