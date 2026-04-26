#include <cstddef>
#include <linalg/stride_view.h>
#include <gtest/gtest.h>

using linalg::StrideIterator;
using linalg::StrideView;

template class StrideIterator<int>;
template class StrideView<int>;

//==============================================================================
// STRIDE ITERATOR
//==============================================================================
class StrideIteratorTest : public testing::Test
{
protected:
    StrideIteratorTest()
    {}
    static constexpr int A_size { 10 };
    int A[A_size] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
};

// Construction (and getStride(), getPtr())
TEST_F(StrideIteratorTest, Constructor)
{
    StrideIterator<int> it(1, A);
    EXPECT_EQ(it.getPtr(), A);
    EXPECT_EQ(it.getStride(), 1);
}

TEST_F(StrideIteratorTest, DereferenceOperator)
{
    StrideIterator<int> it0 (1, A);
    StrideIterator<int> itN (1, A+A_size-1);
    EXPECT_EQ(*it0, A[0]);
    EXPECT_EQ(*itN, A[A_size-1]);
}

TEST_F(StrideIteratorTest, PlusIntegralOperator)
{
    StrideIterator<int> it1 (1, A);
    StrideIterator<int> it2 (2, A);

    EXPECT_EQ(*(it1 + (A_size - 1)), A[A_size-1]);
    EXPECT_EQ(*(it2 + 1), A[2]);
}

TEST_F(StrideIteratorTest, MinusIntegralOperator)
{
    StrideIterator<int> it1 (1, A + A_size - 1);
    StrideIterator<int> it2 (2, A + A_size - 1);

    EXPECT_EQ(*(it1 - (A_size - 1)), A[0]);
    EXPECT_EQ(*(it2 - 1), A[A_size-3]);
}

TEST_F(StrideIteratorTest, MinusIteratorOperator)
{
    StrideIterator<int> it1 (1, A);
    StrideIterator<int> it2 (1, A + A_size);

    EXPECT_EQ(it2 - it1, A_size);
}

TEST_F(StrideIteratorTest, PreIncrementOperator)
{
    StrideIterator<int> it1 (1, A);
    StrideIterator<int> it2 (2, A);
    ++it1;
    ++it2;
    EXPECT_EQ(*it1, A[1]);
    EXPECT_EQ(*it2, A[2]);
}

TEST_F(StrideIteratorTest, PostIncrementOperator)
{
    StrideIterator<int> it1 (1, A);
    StrideIterator<int> it2 (2, A);

    EXPECT_EQ(*(it1++), A[0]);
    EXPECT_EQ(*(it2++), A[0]);
    EXPECT_EQ(*it1, A[1]);
    EXPECT_EQ(*it2, A[2]);
}

TEST_F(StrideIteratorTest, PreDecrementOperator)
{
    StrideIterator<int> it1 (1, A + A_size - 1);
    StrideIterator<int> it2 (2, A + A_size - 1);
    --it1;
    --it2;
    EXPECT_EQ(*it1, A[A_size - 2]);
    EXPECT_EQ(*it2, A[A_size - 3]);
}

TEST_F(StrideIteratorTest, PostDecrementOperator)
{
    StrideIterator<int> it1 (1, A + A_size - 1);
    StrideIterator<int> it2 (2, A + A_size - 1);

    EXPECT_EQ(*(it1--), A[A_size - 1]);
    EXPECT_EQ(*(it2--), A[A_size - 1]);
    EXPECT_EQ(*it1, A[A_size - 2]);
    EXPECT_EQ(*it2, A[A_size - 3]);
}

TEST_F(StrideIteratorTest, BracketOperator)
{
    StrideIterator<int> it1 (1, A);
    StrideIterator<int> it2 (2, A);

    EXPECT_EQ(it1[0], A[0]);
    EXPECT_EQ(it1[A_size-1], A[A_size-1]);

    EXPECT_EQ(it2[0], A[0]);
    EXPECT_EQ(it2[1], A[2]);
}

TEST_F(StrideIteratorTest, ComparisonOperatorIterator)
{
    StrideIterator<int> it1 (1, A);
    StrideIterator<int> it2 (1, A + A_size - 1);
    StrideIterator<int> it3 (2, A); // Same a 1 but different stride

    EXPECT_TRUE((it1 <=> it2) == std::weak_ordering::less);
    EXPECT_TRUE((it2 <=> it1) == std::weak_ordering::greater);
    EXPECT_TRUE((it1 <=> it3) == std::weak_ordering::equivalent);

    EXPECT_TRUE(it1 == it1);
    EXPECT_TRUE(it1 < it2);
}

TEST_F(StrideIteratorTest, ComparisonOperatorPointer)
{
    StrideIterator<int> it1 (1, A);
    StrideIterator<int> it2 (1, A + A_size - 1);
    StrideIterator<int> it3 (2, A); // Same a 1 but different stride

    EXPECT_TRUE((A <=> it2) == std::weak_ordering::less);
    EXPECT_TRUE((it2 <=> A) == std::weak_ordering::greater);
    EXPECT_TRUE((it3 <=> A) == std::weak_ordering::equivalent);

    EXPECT_TRUE(A == it1);
    EXPECT_TRUE(A < it2);
}

//==============================================================================
// STRIDE ITERATOR
//==============================================================================
class StrideViewTest : public testing::Test
{
protected:
    StrideViewTest()
    {}
    static constexpr int A_size { 10 };
    int A[A_size] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
};

TEST_F(StrideViewTest, Construction)
{
    StrideView<int> sv(A, 1, A_size);
    EXPECT_EQ(sv.getPtr(), A);
    EXPECT_EQ(sv.getStride(), 1);
    EXPECT_EQ(sv.getCount(), A_size);
}

TEST_F(StrideViewTest, Begin)
{
    StrideView<int> sv(A, 1, A_size);
    EXPECT_EQ(sv.begin(), A);
}

TEST_F(StrideViewTest, End)
{
    StrideView<int> sv1(A, 1, A_size);
    EXPECT_EQ(sv1.end(), A+A_size);

    StrideView<int>::size_type str { 3 };
    StrideView<int>::size_type count { (A_size - 1)/str + 1 };
    StrideView<int> sv2(A, str, count);
    EXPECT_EQ(sv2.end(), A + str*count);
}

TEST_F(StrideViewTest, BracketOperator)
{
    StrideView<int> sv1(A, 1, A_size);
    EXPECT_EQ(sv1[0], A[0]);
    EXPECT_EQ(sv1[A_size-1], A[A_size-1]);
    sv1[1] = 99;
    EXPECT_EQ(A[1], 99);

    StrideView<int>::size_type str { 3 };
    StrideView<int>::size_type count { (A_size - 1)/str + 1 };
    StrideView<int> sv2(A, str, count);
    EXPECT_EQ(sv2[0], A[0]);
    EXPECT_EQ(sv2[1], A[str]);
}