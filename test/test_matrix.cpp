#include "gtest/gtest.h"
#include <linalg/Matrix.h>

TEST(ArrayTest, TestMembers)
{
    linalg::Matrix<int> mat { 1, 5, 2 };
    EXPECT_EQ(mat.rows(), 1);
    EXPECT_EQ(mat.cols(), 5);
}