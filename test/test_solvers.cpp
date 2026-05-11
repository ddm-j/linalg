#include <algorithm>
#include <cstddef>
#include <climits>
#include <linalg/errors.h>
#include <linalg/stride_view.h>
#include <linalg/Matrix.h>
#include <linalg/solvers.h>
#include <gtest/gtest.h>

using linalg::Matrix;
using linalg::StrideView;
using namespace linalg::solvers;

//==============================================================================
// Utilities
//==============================================================================
template <typename T>
void test_ranges_equal(const StrideView<T>& v1, const StrideView<T>& v2, int fac=10)
{
    T eps { fac * std::numeric_limits<T>::epsilon() };

    ASSERT_EQ(v1.getCount(), v2.getCount()) << "Ranges have different sizes.";
    for (auto i {0}; i < v1.getCount(); ++i)
    {
        EXPECT_NEAR(v1[i], v2[i], eps) << "Ranges differ at i = " << i;
    }
}


//==============================================================================
// Forward Sub
//==============================================================================
TEST(ForwardSub, NonLowerTriThrows)
{
    Matrix<double> A {
        { 4, 0, 1 },
        { 1, 5, 0 },
        { 1, 2, 3 }
    };
    Matrix<double> b(3, 1, {1, 2, 3});
    EXPECT_THROW((forward_sub(A, b)), linalg::NonLowerTriangular);
}

TEST(ForwardSub, IsCorrect)
{
    Matrix<double> A {
        { 1, 0, 0 },
        { 2, 5, 0 },
        { 3, 9, 1 }
    };
    Matrix<double> x(3, 1, {1, 2, 3});
    auto b { A*x };
    auto x_sol { forward_sub(A, b) };

    // Shape Check
    EXPECT_EQ(x.rows(), x_sol.rows());
    EXPECT_EQ(x.cols(), x_sol.cols());

    // Result Check
    test_ranges_equal(x.colit(0), x_sol.colit(0));
}

//==============================================================================
// Backward Sub
//==============================================================================
TEST(BackwardSub, NonUpperTriThrows)
{
    Matrix<double> A {
        { 4, 1, 9 },
        { 0, 5, 7 },
        { 1, 0, 3 }
    };
    Matrix<double> b(3, 1, {1, 2, 3});
    EXPECT_THROW((backward_sub(A, b)), linalg::NonUpperTriangular);
}

TEST(BackwardSub, IsCorrect)
{
    Matrix<double> A {
        { 1, 2, 3 },
        { 0, 5, 9 },
        { 0, 0, 1 }
    };
    Matrix<double> x(3, 1, {1, 2, 3});
    auto b { A*x };
    auto x_sol { backward_sub(A, b) };

    // Shape Check
    EXPECT_EQ(x.rows(), x_sol.rows());
    EXPECT_EQ(x.cols(), x_sol.cols());

    // Result Check
    test_ranges_equal(x.colit(0), x_sol.colit(0));
}

//==============================================================================
// LU Solver
//==============================================================================
TEST(LUSolver, NonSquareMatrixThrows)
{
    Matrix<double> A {
        { 4, 1, 2 },
        { 1, 5, 1 },
    };
    Matrix<double> b(3, 1, {1, 2, 3});
    EXPECT_THROW((solve_lu(A, b)), linalg::ShapeError);
}

TEST(LUSolver, VectorLengthMismatchThrows)
{
    Matrix<double> A {
        { 4, 1, 2 },
        { 1, 5, 1 },
        { 1, 2, 3 }
    };
    Matrix<double> b(4, 1, {1, 2, 3, 4});
    EXPECT_THROW((solve_lu(A, b)), linalg::ShapeError);
}

TEST(LUSolver, VectorIsMatrixThrows)
{
    Matrix<double> A {
        { 4, 1, 2 },
        { 1, 5, 1 },
        { 1, 2, 3 }
    };
    Matrix<double> b(3, 2, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW((solve_lu(A, b)), linalg::ShapeError);
}

TEST(LUSolver, NoPivot)
{
    Matrix<double> A1 {
        { 4, 1, 2 },
        { 1, 5, 1 },
        { 2, 1, 6 }
    };
    Matrix<double> x(3, 1, {1, 2, 3});
    auto x_sol { solve_lu(A1, A1*x) };

    // Shape Check
    EXPECT_EQ(x.rows(), x_sol.rows());
    EXPECT_EQ(x.cols(), x_sol.cols());

    // Result Check
    test_ranges_equal(x.colit(0), x_sol.colit(0));
}

TEST(LUSolver, Pivot)
{
    Matrix<double> A2 {
        { 0, 1, 2 },
        { 1, 3, 1 },
        { 2, 1, 4 }
    };
    Matrix<double> x(3, 1, {1, 2, 3});
    auto x_sol { solve_lu(A2, A2*x) };

    // Shape Check
    EXPECT_EQ(x.rows(), x_sol.rows());
    EXPECT_EQ(x.cols(), x_sol.cols());

    // Result Check
    test_ranges_equal(x.colit(0), x_sol.colit(0));
}

TEST(LUSolver, Singular)
{
    Matrix<double> A3 {
        { 1, 2, 3 },
        { 2, 4, 6 },
        { 1, 1, 1 }
    };
    Matrix<double> x(3, 1, {1, 2, 3});
    EXPECT_THROW((solve_lu(A3, A3*x)), linalg::Singular);
}

TEST(LUSolver, Identity)
{
    auto I { Matrix<double>::eye(3, 3) };
    Matrix<double> b(3, 1, {1, 2, 3});
    auto b_res { solve_lu(I, b) };

    // Shape Check
    EXPECT_EQ(b.rows(), b_res.rows());
    EXPECT_EQ(b.cols(), b_res.cols());

    // Result Check
    test_ranges_equal(b.colit(0), b_res.colit(0));
}

//==============================================================================
// Cholesky Solver
//==============================================================================
TEST(CholeskySolver, NonSquareMatrixThrows)
{
    Matrix<double> A {
        { 4, 1, 2 },
        { 1, 5, 1 },
    };
    Matrix<double> b(3, 1, {1, 2, 3});
    EXPECT_THROW((solve_cholesky(A, b)), linalg::ShapeError);
}

TEST(CholeskySolver, VectorLengthMismatchThrows)
{
    Matrix<double> A {
        { 4, 1, 2 },
        { 1, 5, 1 },
        { 1, 2, 3 }
    };
    Matrix<double> b(4, 1, {1, 2, 3, 4});
    EXPECT_THROW((solve_cholesky(A, b)), linalg::ShapeError);
}

TEST(CholeskySolver, VectorIsMatrixThrows)
{
    Matrix<double> A {
        { 4, 1, 2 },
        { 1, 5, 1 },
        { 1, 2, 3 }
    };
    Matrix<double> b(3, 2, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW((solve_cholesky(A, b)), linalg::ShapeError);
}

TEST(CholeskySolver, Identity)
{
    auto I { Matrix<double>::eye(3, 3) };
    Matrix<double> b(3, 1, {1, 2, 3});
    auto b_res { solve_cholesky(I, b) };

    // Shape Check
    EXPECT_EQ(b.rows(), b_res.rows());
    EXPECT_EQ(b.cols(), b_res.cols());

    // Result Check
    test_ranges_equal(b.colit(0), b_res.colit(0));
}

TEST(CholeskySolver, IsCorrect)
{
    Matrix<double> A {
        {6, 3, 4, 8},
        {3, 6, 5, 1},
        {4, 5,10, 7},
        {8, 1, 7,25}
    };
    Matrix<double> x(4, 1, {1, 2, 3, 4});
    auto x_sol { solve_cholesky(A, A*x) };

    // Shape Check
    EXPECT_EQ(x.rows(), x_sol.rows());
    EXPECT_EQ(x.cols(), x_sol.cols());

    // Result Check
    test_ranges_equal(x.colit(0), x_sol.colit(0));
}