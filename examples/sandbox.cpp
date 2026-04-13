#include <iostream>
#include <linalg/Matrix.h>
#include <linalg/stride_view.h>

int main()
{
    linalg::Matrix<int> mat {
        { 1, 5, 2, 10, 35, 50 },
        { 3, 9, 4,  1,  6,  8 },
        { 6, 4,-1,  2,  8,  3 },
        { 4, 9, 9,  5,  4,  2 }
    };
    linalg::Matrix<int> ones { linalg::Matrix<int>::ones(3, 3) };
    linalg::Matrix<int> eye { linalg::Matrix<int>::eye(10) };

    std::cout << eye << "\n";

    auto row1 { mat.rowit(1) };
    std::cout << "Row 1: ";
    for (auto& e : row1)
    {
        std::cout << e << " ";
    }
    std::cout << "\n";

    auto col1 { mat.colit(1) };
    std::cout << "Col 1: ";
    for (auto& e : col1)
    {
        std::cout << e << " ";
    }
    std::cout << "\n";

    auto diag { mat.diagit() };
    std::cout << "Diagonal: ";
    for (auto& e : diag)
    {
        std::cout << e << " ";
    }
    std::cout << "\n";
    // std::cout << "\n";
    // int x { 69 };
    // int arr[6] {1, 2, 3, 4, 5, 6}; 

    // std::ptrdiff_t n { 6 };
    // std::ptrdiff_t stride { 2 };
    // std::ptrdiff_t count { (n - 1) / stride + 1 };

    // linalg::StrideView<int> view(arr, stride, count);

    // for (const auto& e : view)
    // {
    //     std::cout << e << " ";
    // }
    // std::cout << "\n";

    // std::cout << "View[0] = " << view[0] << "\n";
    // std::cout << "View[1] = " << view[1] << "\n";
    // std::cout << "View[2] = " << view[2] << "\n";

    return 0;
}