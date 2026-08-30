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

template <typename T>
void test_matrices_approx_equal(const Matrix<T>& A, const Matrix<T>& B, T fac = T{10}, T n = T {3})
{
    // A = Expected Value, B = Testing Value
    using std::abs;
    using std::max;
    using size_type = Matrix<T>::size_type;

    // Tolerance Setup
    T eps { std::numeric_limits<T>::epsilon() };
    T rtol { eps * fac * n };
    T atol { eps * fac * n * (*(std::max_element(A.begin(), A.end()))) };

    ASSERT_EQ(A.rows(), B.rows()) << "Matrix A and B have different number of rows.";
    ASSERT_EQ(A.cols(), B.cols()) << "Matrix A and B have different numer of cols.";
    for (size_type i {}; i < A.length(); ++i)
    {
        EXPECT_LE(abs(A[i]-B[i]), atol + rtol * max(abs(A[i]),abs(B[i]))) << 
                    std::format("Matrices differ at [{}, {}]", A.ridx(i), B.cidx(i));
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

//==============================================================================
// Gauss Jordan Solver
//==============================================================================
TEST(GaussJordan, NonSquareThrows)
{
    Matrix<double> A {
        { 2, 1, 1, 1 },
        { 1, 3, 2, 1 },
        { 1, 0, 0, 1 }
    };
    Matrix<double> B {
        { 3, 2 },
        { 5, 3 },
        { 1, 0 }
    };

    // Solutions
    EXPECT_THROW(solve_gauss_jordan(A, B), linalg::ShapeError);
}

TEST(GaussJordan, ColsAvsRowsB)
{
    Matrix<double> A {
        { 2, 1, 1 },
        { 1, 3, 2 },
        { 1, 0, 0 }
    };
    Matrix<double> B {
        { 3, 2 },
        { 5, 3 },
        { 1, 0 },
        { 1, 0}
    };

    // Solutions
    EXPECT_THROW(solve_gauss_jordan(A, B), linalg::ShapeError);
}

TEST(GaussJordan, CorrectnessOnKnown)
{
    Matrix<double> A {
        { 2, 1, 1 },
        { 1, 3, 2 },
        { 1, 0, 0 }
    };
    Matrix<double> B {
        { 3, 2 },
        { 5, 3 },
        { 1, 0 }
    };

    // Solutions
    Matrix<double> A_inv_expected {
        {  0,  0,  1 },
        { -2,  1,  3 },
        {  3, -1, -5 }
    };

    Matrix<double> X_expected {
        {  1,  0 },
        {  2, -1 },
        { -1,  3 }
    };

    solve_gauss_jordan(A, B);
    
    // Tests
    test_matrices_approx_equal(A_inv_expected, A);
    test_matrices_approx_equal(X_expected, B);
}

TEST(GaussJordan, PivotPath)
{
    Matrix<double> A {
        { 0, 1, 2 },
        { 1, 0, 1 },
        { 2, 1, 0 }
    };
    
    Matrix<double> B {
        { 4, 1 },
        { 3, 3 },
        { 2, 3 }
    };
    // Solutions
    Matrix<double> A_inv_expected {
        { -0.25,  0.5 ,  0.25 },
        {  0.5 , -1.0 ,  0.5  },
        {  0.25,  0.5 , -0.25 }
    };
    
    Matrix<double> X_expected {
        { 1,  2 },
        { 0, -1 },
        { 2,  1 }
    };

    solve_gauss_jordan(A, B);
    
    // Tests
    test_matrices_approx_equal(A_inv_expected, A);
    test_matrices_approx_equal(X_expected, B);
}

TEST(GaussJordan, RoundTripResidual)
{
    Matrix<double> A {
        { 0, 1, 2 },
        { 1, 0, 1 },
        { 2, 1, 0 }
    };
    auto A_orig = A;

    Matrix<double> B {
        { 4, 1 },
        { 3, 3 },
        { 2, 3 }
    };
    auto B_orig = B;
    Matrix<double> I { Matrix<double>::eye(3, 3) };

    // Solutions
    solve_gauss_jordan(A, B);
    
    // Tests
    SCOPED_TRACE("A_orig * B == B_orig");
    test_matrices_approx_equal(A_orig*B, B_orig);
    SCOPED_TRACE("A_orig * A == I");
    test_matrices_approx_equal(A_orig*A, I);
    SCOPED_TRACE("A * A_orig == I");
    test_matrices_approx_equal(A*A_orig, I);
}

TEST(GaussJordan, Singular)
{
    Matrix<double> A_sing {
        { 1, 2, 3 },
        { 2, 4, 6 },
        { 1, 0, 1 }
    };

    Matrix<double> B_sing(3, 1, {1, 2, 1});

    // Test
    EXPECT_THROW(solve_gauss_jordan(A_sing, B_sing), linalg::Singular);
}

TEST(GaussJordan, NearSingular)
{
    Matrix<double> A_near {
        { 1, 2, 3 },
        { 2, 4 + 1e-16, 6 },
        { 1, 0, 1 }
    };

    Matrix<double> B_near(3, 1, {1, 2, 1});

    // Test
    EXPECT_THROW(solve_gauss_jordan(A_near, B_near), linalg::Singular);
}

TEST(GaussJordan, Conditioning)
{
    // J with columns 2 and 3 nearly collinear
    Matrix<double> JtJ {
        { 2.0,       3.0,       3.0       },
        { 3.0,       5.0,       5.0 - 1e-8 },
        { 3.0,       5.0 - 1e-8, 5.0      }
    };
    Matrix<double> g(3, 1, {1.0, 1.0, 1.0});

    const double lambda_large = 1.0;    // heavily damped
    const double lambda_small = 1e-12;  // essentially undamped
    Matrix<double> I {Matrix<double>::eye(3, 3) };

    // Test Case 1:
    Matrix<double> A1 { JtJ + lambda_large*I };
    auto A1_orig = A1;
    auto X1 = g;
    solve_gauss_jordan(A1, X1);

    SCOPED_TRACE("A1_orig * X == g");
    test_matrices_approx_equal(A1_orig*X1, g);



    // Test Case 2:
    Matrix<double> A2 { JtJ + lambda_small*I };
    auto A2_orig = A2;
    auto X2 = g;

    solve_gauss_jordan(A2, X2);

    SCOPED_TRACE("A2_orig * X2 == g");
    test_matrices_approx_equal(A2_orig*X2, g);

    // Check Relative Values
    double X1_norm {
        *(std::max_element(X1.begin(), X1.end(), 
                [](double a, double b) {
                    return std::abs(a) > std::abs(b);
                }
            )
        )
    };
    double X2_norm {
        *(std::max_element(X2.begin(), X2.end(), 
                [](double a, double b) {
                    return std::abs(a) > std::abs(b);
                }
            )
        )
    };
    EXPECT_GT(X2_norm / X1_norm, 1e4);
}

// 5. LM-representative conditioning
// A small JᵀJ + λI with a genuinely ill-conditioned JᵀJ, checked at two λ values: large λ gives an accurate solve, λ→0 degrades. Documents the actual behavior you'll depend on.