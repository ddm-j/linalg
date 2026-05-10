#include <cstddef>
#include <climits>
#include <linalg/Matrix.h>
#include <gtest/gtest.h>

using linalg::Matrix;

// Insantiate for Coverage
template class Matrix<int>;
template class Matrix<double>;

//==============================================================================
// CONSTRUCTION
//==============================================================================
TEST(Construction, DimensionConstructor)
{
    Matrix<int> m(5, 5);
    EXPECT_EQ(m.rows(), 5);
    EXPECT_EQ(m.cols(), 5);
    EXPECT_EQ(m.length(), 25);
}

TEST(Construction, OverflowThrows)
{
    EXPECT_THROW((Matrix<int>(SIZE_MAX, 2)), std::bad_array_new_length);
}

TEST(Construction, SingleListConstructor)
{
    Matrix<int> m { 1, 2, 3, 4, 5 };
    EXPECT_EQ(m.rows(), 1);
    EXPECT_EQ(m.cols(), 5);
    EXPECT_EQ(m.length(), 5);
    EXPECT_EQ(m[4], 5);
}

TEST(Construction, StackedListConstructor)
{
    Matrix<int> m {
        { 1, 2, 3, 4, 5 },
        { 6, 7, 8, 9, 10}
    };
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 5);
    EXPECT_EQ(m.length(), 10);
    EXPECT_EQ(m[9], 10);
    EXPECT_EQ((m[0,4]), 5);
}

TEST(Construction, JaggedListThrows)
{
    EXPECT_THROW((Matrix<int>{{ 1, 2, 3, 4, 5 },{ 6, 7, 8, 9}}), std::invalid_argument);
}

TEST(Construction, DimsListConstructor)
{
    Matrix<int> m(2, 5, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 5);
    EXPECT_EQ(m.length(), 10);
    EXPECT_EQ(m[9], 10);
    EXPECT_EQ((m[0,4]), 5);
}

TEST(Construction, DimsListConstructorThrows)
{
    EXPECT_THROW((Matrix<int>(2, 5, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})), std::invalid_argument);
}
//==============================================================================
// SEMANTICS
//==============================================================================
TEST(Semantics, CopyConstructor)
{
    Matrix<int> a {{1, 2}, {3, 4}};
    Matrix<int> copy{a};
    for (Matrix<int>::size_type i = 0; i < a.length(); ++i)
        EXPECT_EQ(copy[i], a[i]);
    
    copy[0] = 99;
    EXPECT_NE(copy[0], a[0]);
}

TEST(Semantics, CopyAssignmentOperator)
{
    Matrix<int> a {{1, 2}, {3, 4}};
    Matrix<int> copy{{1, 2}};
    copy = a;
    for (Matrix<int>::size_type i = 0; i < a.length(); ++i)
        EXPECT_EQ(copy[i], a[i]);
    
    copy[0] = 99;
    EXPECT_NE(copy[0], a[0]);
}

TEST(Semantics, MoveConstructor)
{
    Matrix<int> a {{1, 2}, {3, 4}};
    Matrix<int> move(std::move(a));

    EXPECT_EQ(move.length(), 4);
    EXPECT_EQ(move[0], 1);
    EXPECT_EQ(a.length(), 0);
    EXPECT_EQ(a.rows(), 0);
    EXPECT_EQ(a.cols(), 0);
}

TEST(Semantics, MoveAssignmentOperator)
{
    Matrix<int> a {{1, 2}, {3, 4}};
    Matrix<int> move { 5, 6 };
    move = std::move(a);

    EXPECT_EQ(move.length(), 4);
    EXPECT_EQ(move[0], 1);
    EXPECT_EQ(a.length(), 0);
    EXPECT_EQ(a.rows(), 0);
    EXPECT_EQ(a.cols(), 0);
}

//==============================================================================
// STATICS
//==============================================================================
TEST(Statics, Ones)
{
    using size_type = Matrix<int>::size_type;
    Matrix<int> ones { Matrix<int>::ones(3, 3) };
    for (size_type i { 0 }; i < ones.length(); ++i)
    {
        EXPECT_EQ(ones[i], 1);
    }
}

TEST(Statics, Eye)
{
    Matrix<int> eye { Matrix<int>::eye(3, 4) };
    EXPECT_EQ((eye[0, 0]), 1);
    EXPECT_EQ((eye[1, 1]), 1);
    EXPECT_EQ((eye[2, 2]), 1);
}

//==============================================================================
// UTILITY
//==============================================================================
TEST(Utility, Accessors)
{
    Matrix<int> a {{1, 2}, {3, 4}};
    EXPECT_EQ(a.rows(), 2);
    EXPECT_EQ(a.cols(), 2);
    EXPECT_EQ(a.ridx(3), 1);
    EXPECT_EQ(a.ridx(3), 1);
    EXPECT_EQ(a.length(), 4);
}

TEST(Utility, BracketOperator)
{
    Matrix<int> a {{1, 2}, {3, 4}};
    
    // Non Const
    // Dual Bracket
    EXPECT_EQ((a[0, 1]), 2);
    a[0, 1] = 3;
    EXPECT_EQ((a[0, 1]), 3);

    // Single Bracket
    EXPECT_EQ((a[1]), 3);
    a[1] = 2;
    EXPECT_EQ((a[1]), 2);
}

TEST(Utility, BracketOperatorConst)
{
    const Matrix<int> a {{1, 2}, {3, 4}};
    
    // Const
    // Dual Bracket
    EXPECT_EQ((a[0, 1]), 2);

    // Single Bracket
    EXPECT_EQ((a[1]), 2);
}

TEST(Utility, Iterators)
{
    Matrix<int> a {{1, 2}, {3, 4}};
    Matrix<int>::size_type l { a.length() };

    EXPECT_EQ(*(a.begin()), 1);
    EXPECT_EQ(*(a.begin()+(l-1)), 4);
    EXPECT_EQ(*(a.end()-1), 4);
    EXPECT_EQ(*(a.end()-l), 1);
}

//==============================================================================
// ARITHMETIC
//==============================================================================
class Arithmetic : public testing::Test
{
protected:
    Arithmetic()
    {
    }
    Matrix<int> A {{1,2},{3,4}};
    Matrix<int> B {{1,2,3,4},{5,6,7,8}};
    Matrix<int> C {{1,2,3},{4,5,6},{7,8,9}};
    Matrix<int> AxB {{11,14,17,20},{23,30,37,44}};
    Matrix<int> Ones {{1,1},{1,1}};
};

// Unary Minus
TEST_F(Arithmetic, UnaryMinus)
{
    Matrix<int> minusA { -A };
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(-A[i], minusA[i]);
}

// Addition Assignment
TEST_F(Arithmetic, AdditionAssignment)
{
    Matrix<int> copy { A };
    A += Ones;
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(copy[i]+1, A[i]);
}

TEST_F(Arithmetic, AdditionAssignmentThrows)
{
    EXPECT_THROW(A+=B, std::invalid_argument);
}

// Subtraction Assignment
TEST_F(Arithmetic, SubtractionAssignment)
{
    Matrix<int> copy { A };
    A -= Ones;
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(copy[i]-1, A[i]);
}

TEST_F(Arithmetic, SubtractionAssignmentThrows)
{
    EXPECT_THROW(A-=B, std::invalid_argument);
}

// Matrix Addition
TEST_F(Arithmetic, MatrixAddition)
{
    Matrix<int> copy { A };
    Matrix<int> plusOnes { A + Ones };
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(copy[i]+1, plusOnes[i]);
}

TEST_F(Arithmetic, MatrixAdditionThrows)
{
    EXPECT_THROW(A+B, std::invalid_argument);
}

// Matrix Subtraction
TEST_F(Arithmetic, MatrixSubtraction)
{
    Matrix<int> copy { A };
    Matrix<int> minusOnes { A - Ones };
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(copy[i]-1, minusOnes[i]);
}

TEST_F(Arithmetic, MatrixSubtractionThrows)
{
    EXPECT_THROW(A-B, std::invalid_argument);
}

// Matrix Multiplication
TEST_F(Arithmetic, MatrixMultiplication)
{
    Matrix<int> AtimesB { A*B };
    EXPECT_EQ(AtimesB.rows(), A.rows());
    EXPECT_EQ(AtimesB.cols(), B.cols());
    for (Matrix<int>::size_type i = 0; i < AxB.length(); ++i)
        EXPECT_EQ(AxB[i], AtimesB[i]);
}

TEST_F(Arithmetic, MatrixMultiplicationThrows)
{
    EXPECT_THROW(A*C, std::invalid_argument);
}

// Scalar Multiplication
TEST_F(Arithmetic, ScalarMultiplication)
{
    int V { 2 };
    // LHS
    Matrix<int> AopV { V * A };
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(A[i]*V, AopV[i]);

    // RHS
    AopV = A * V;
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(A[i]*V, AopV[i]);
}

// Scalar Division
TEST_F(Arithmetic, ScalarDivision)
{
    int V { 2 };
    Matrix<int> AopV { A / V };
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(A[i]/V, AopV[i]);
}

TEST_F(Arithmetic, ScalarDivisionThrows)
{
    EXPECT_THROW(A / 0, std::invalid_argument);
}

// Scalar Addition
TEST_F(Arithmetic, ScalarAddition)
{
    int V { 2 };
    // LHS
    Matrix<int> AopV { V + A };
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(A[i]+V, AopV[i]);

    // RHS
    AopV = A + V;
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(A[i]+V, AopV[i]);
}

// Scalar Subtraction
TEST_F(Arithmetic, ScalarSubtraction)
{
    int V { 2 };
    // LHS
    Matrix<int> AopV { V - A };
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(V-A[i], AopV[i]);

    // RHS
    AopV = A - V;
    for (Matrix<int>::size_type i = 0; i < A.length(); ++i)
        EXPECT_EQ(A[i]-V, AopV[i]);
}

//==============================================================================
// VIEWS
//==============================================================================
class View : public testing::Test
{
protected:
    View()
    {
    }
    Matrix<int> A {{1,2},{3,4}};
    const Matrix<int> B {{1,2},{3,4}};

    Matrix<int> C {{1,2,3},{4,5,6}};
    const Matrix<int> D {{1,2,3},{4,5,6}};
};

TEST_F(View, Rows)
{
    auto v { A.rowit(0) };
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    v[1] = 3;
    EXPECT_EQ(v[1], 3);

    auto vConst { B.rowit(0) };
    EXPECT_EQ(vConst[0], 1);
    EXPECT_EQ(vConst[1], 2);
}

TEST_F(View, Columns)
{
    auto v { A.colit(0) };
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 3);
    v[1] = 5;
    EXPECT_EQ(v[1], 5);

    auto vConst { B.colit(0) };
    EXPECT_EQ(vConst[0], 1);
    EXPECT_EQ(vConst[1], 3);
}

TEST_F(View, Diagonal)
{
    auto v { C.diagit() };
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 5);
    v[1] = 3;
    EXPECT_EQ(v[1], 3);

    auto vConst { D.diagit() };
    EXPECT_EQ(vConst[0], 1);
    EXPECT_EQ(vConst[1], 5);
}

//==============================================================================
// NON MEMBER - NON FRIEND UTILTIES
//==============================================================================
TEST(Utilities, IsLowerTriangular)
{
    // Test True
    Matrix<double> A {
        {1, 0, 99*std::numeric_limits<double>::epsilon()},
        {0.5, 0, 0},
        {0.2, -0.1, 1}
    };
    EXPECT_TRUE((linalg::is_lowertri(A,100)));

    // Test False
    Matrix<double> B {
        {1, 0, 101*std::numeric_limits<double>::epsilon()},
        {0.5, 0, 0},
        {0.2, -0.1, 1}
    };
    EXPECT_FALSE((linalg::is_lowertri(B,100)));
}

TEST(Utilities, IsUpperTriangular)
{
    // Test True
    Matrix<double> A {
        {1, -0.1, 2.0},
        {0, 0, 0.5},
        {99*std::numeric_limits<double>::epsilon(), 0, 1}
    };
    EXPECT_TRUE((linalg::is_uppertri(A,100)));

    // Test False
    Matrix<double> B {
        {1, -0.1, 2.0},
        {0, 0, 0.5},
        {101*std::numeric_limits<double>::epsilon(), 0, 1}
    };
    EXPECT_FALSE((linalg::is_uppertri(B,100)));
}