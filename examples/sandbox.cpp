#include <iostream>
#include <linalg/Matrix.h>

int main()
{
    linalg::Matrix<int> mat {
        { 1, 5, 2 },
        { 3, 9, 4 } 
    };
    std::cout << "Testing...\n";
    std::cout << mat << "\n";
    return 0;
}