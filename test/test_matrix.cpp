#include "gtest/gtest.h"
#include <linalg/Matrix.h>

TEST(ArrayTest, TestMembers)
{
    linalg::Matrix mat { 1, 5 };
    EXPECT_EQ(mat.rows(), 1);
    EXPECT_EQ(mat.cols(), 5);
}