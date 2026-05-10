#include <algorithm>
#include <cstddef>
#include <climits>
#include <linalg/errors.h>
#include <linalg/Matrix.h>
#include <linalg/decomposition.h>
#include <gtest/gtest.h>

using linalg::Matrix;
using namespace linalg::decomposition;

//==============================================================================
// LU Decomposition
//==============================================================================
TEST(LU, NoPivot)
{
    using size_type = Matrix<double>::size_type;

    Matrix<double> A {
        { 4, 1, 2 },
        { 1, 5, 1 },
        { 2, 1, 6 }
    };
    auto [P, L, U] { lu(A)};
    auto res { L*U };

    // Multiplication is Correct
    for (size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(A[i], res[i]);
}

TEST(LU, Pivot)
{
    using size_type = Matrix<double>::size_type;

    Matrix<size_type> P_solution(3, 1, {2, 1, 0});
    Matrix<double> A {
        { 0, 1, 2 },
        { 1, 3, 1 },
        { 2, 1, 4 }
    };
    auto [P, L, U] { lu(A)};
    auto res { L*U };

    // Permutation is Correct
    for (size_type i = 0; i < P.length(); ++i)
        EXPECT_EQ(P[i], P_solution[i]);

    // Multiplication is Correct
    for (size_type i = 0; i < A.rows(); ++i)
        EXPECT_TRUE((std::ranges::equal(res.rowit(i), A.rowit(P[i]))));
}

TEST(LU, Structures1x1)
{
    using size_type = Matrix<double>::size_type;

    // 1x1
    Matrix<double> A(1, 1, {2});
    auto [P, L, U] { lu(A) };
    auto res { L * U };
    for (size_type i = 0; i < A.rows(); ++i)
        EXPECT_TRUE((std::ranges::equal(res.rowit(i), A.rowit(P[i]))));
}

TEST(LU, Structures2x2)
{
    using size_type = Matrix<double>::size_type;

    // 2x2
    Matrix<double> A {
        { 3, 7 },
        { 6, 1 }
    };
    auto [P, L, U] { lu(A) };
    auto res { L * U };
    for (size_type i = 0; i < A.rows(); ++i)
        EXPECT_TRUE((std::ranges::equal(res.rowit(i), A.rowit(P[i]))));
}

TEST(LU, StructuresTall)
{
    using size_type = Matrix<double>::size_type;

    // 4x2
    Matrix<double> A {
        { 3, 7 },
        { 6, 1 },
        { 3, 4 },
        { 7, 9 }
    };
    auto [P, L, U] { lu(A) };
    auto res { L * U };
    for (size_type i = 0; i < A.rows(); ++i)
        EXPECT_TRUE((std::ranges::equal(res.rowit(i), A.rowit(P[i]))));
}

TEST(LU, StructuresWide)
{
    using size_type = Matrix<double>::size_type;

    // 4x2
    Matrix<double> A {
        { 3, 6, 3, 7},
        { 7, 1, 4, 9}
    };
    auto [P, L, U] { lu(A) };
    auto res { L * U };
    for (size_type i = 0; i < A.rows(); ++i)
        EXPECT_TRUE((std::ranges::equal(res.rowit(i), A.rowit(P[i]))));
}

TEST(LU, SingularMatrix)
{
    Matrix<double> singular {
        { 1, 2, 3 },
        { 2, 4, 6 },
        { 1, 1, 1 }
    };
    EXPECT_THROW(lu(singular), linalg::Singular);
}

//==============================================================================
// Cholesky Decomposition
//==============================================================================
TEST(Cholesky, NonSquare)
{
    Matrix<double> rect {
        { 1, 2 },
        { 2, 4 },
        { 1, 1 }
    };
    EXPECT_THROW(cholesky(rect), linalg::ShapeError);
}

TEST(Cholesky, NonSymmetric)
{
    Matrix<double> nonsym {
        { 1, 0, 1 },
        { 0, 1, 0 },
        { 0, 0, 1 }
    };
    EXPECT_THROW(cholesky(nonsym), linalg::NonSymmetric);
}

TEST(Cholesky, Indefinite)
{
    Matrix<double> indef {
        { 4, 2, 2 },
        { 2, 2, 1 },
        { 2, 1,-5 }
    };
    EXPECT_THROW(cholesky(indef), linalg::Indefinite);
}

TEST(Cholesky, Singular)
{
    Matrix<double> sing {
        { 1, 1, 0 },
        { 1, 1, 0 },
        { 0, 0, 1 }
    };
    EXPECT_THROW(cholesky(sing), linalg::Singular);
}

TEST(Cholesky, IsCorrect)
{
    Matrix<double> A {
        {6, 3, 4, 8},
        {3, 6, 5, 1},
        {4, 5,10, 7},
        {8, 1, 7,25}
    };
    auto L { cholesky(A) };
    auto A_t { L * L.transpose() };

    double eps { 10 * std::numeric_limits<double>::epsilon() };

    for (Matrix<double>::size_type i {0}; i < A_t.length(); ++i)
    {
        EXPECT_NEAR(A[i], A_t[i], eps) << "Ranges differ at i = " << i;
    }
}