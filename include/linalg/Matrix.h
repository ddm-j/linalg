#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <memory>
#include <iterator>
#include <iostream>

namespace linalg {

// Forward Declarations
class Matrix;
bool incompatibleDims(const Matrix& a, const Matrix& b, bool mult = false);
Matrix operator+(const Matrix& lhs, const Matrix& rhs);
Matrix operator-(const Matrix& lhs, const Matrix& rhs);
Matrix operator*(const Matrix& lhs, const Matrix& rhs);
Matrix operator*(double v, const Matrix& rhs);
Matrix operator*(const Matrix& lhs, double v);
Matrix operator/(const Matrix& lhs, double v);
Matrix operator+(double v, const Matrix& rhs);
Matrix operator+(const Matrix& lhs, double v);
Matrix operator-(double v, const Matrix& rhs);
Matrix operator-(const Matrix& lhs, double v);

class Matrix
{
public:
    using size_type = std::size_t;

    // Constructor
    Matrix(size_type rows, size_type cols)
        : m_arr { std::make_unique<double[]>(rows * cols) }
        , m_length { rows*cols }
        , m_rows { rows }
        , m_cols { cols }
    {}

    // RAII
    // // Destructor
    ~Matrix()
    {}
    // // Copy Constructor
    Matrix(const Matrix& mat)
        : m_arr { std::make_unique<double[]>(mat.m_rows * mat.m_cols) }
        , m_length { mat.m_length }
        , m_rows { mat.m_rows }
        , m_cols { mat.m_cols }
    {
        std::cout << "Copy Constructor\n";
        std::copy_n(mat.begin(), mat.m_length, begin());
    }
    // // Copy Assignment Operator
    Matrix& operator=(const Matrix& mat)
    {
        std::cout << "Copy Assignment\n";
        if (&mat == this)
            return *this;
        
        // Allow copy if length is the same
        if (m_length != mat.m_length)
            m_arr = std::make_unique<double[]>(mat.m_length);

        std::copy_n(mat.begin(), mat.m_length, begin());
        m_length = mat.m_length;
        m_rows = mat.m_rows;
        m_cols = mat.m_cols;
        return *this;
    }
    // // Move Constructor
    Matrix(Matrix&& mat) noexcept
        : m_arr { std::move(mat.m_arr) }
        , m_length { mat.m_length }
        , m_rows { mat.m_rows }
        , m_cols { mat.m_cols }
    {
        std::cout << "Move Constructor\n";
        // Set Empty State
        mat.m_length = 0;
        mat.m_rows = 0;
        mat.m_cols = 0;
    }

    // // Move Assignment Operator
    Matrix& operator=(Matrix&& mat) noexcept
    {
        std::cout << "Move Assignment\n";
        if (&mat == this)
            return *this;
        m_length = mat.m_length;
        m_rows = mat.m_rows;
        m_cols = mat.m_cols;
        m_arr = std::move(mat.m_arr); 

        // Set Empty State
        mat.m_length = 0;
        mat.m_rows = 0;
        mat.m_cols = 0;
        return *this;
    }

    // Accessors
    size_type rows() const { return m_rows; }
    size_type cols() const { return m_cols; }
    size_type ridx(size_type lidx) const { return lidx / m_cols; }
    size_type cidx(size_type lidx) const { return lidx % m_cols; }
    size_type length() const { return m_length; }

    // Member Functions
    const double& at(size_type i, size_type j) const { return m_arr[i * m_cols + j]; }
    double& at(size_type i, size_type j) { return m_arr[i * m_cols + j]; }
    double* begin() { return m_arr.get(); }
    double* end() { return begin() + m_length; }
    const double* begin() const { return m_arr.get(); }
    const double* end() const { return begin() + m_length; }

    // Operator Overloads
    double& operator[](size_type i, size_type j) { return m_arr[i * m_cols + j]; }
    const double& operator[](size_type i, size_type j) const { return m_arr[i * m_cols + j]; }
    Matrix operator-() const { return operator*(-1.0, *this); }
    Matrix& operator+=(const Matrix& rhs)
    {
        if (incompatibleDims(*this, rhs))
        {
            std::cerr << "Matrix addition with incompatible dimensions.\n";
            return *this;
        }
        std::transform(
            begin(), end(),
            rhs.begin(),
            begin(),
            [] (const auto& a, const auto& b)
            {
                return a + b;
            }
        );
        return *this;
    }
    Matrix& operator-=(const Matrix& rhs)
    {
        if (incompatibleDims(*this, rhs))
        {
            std::cerr << "Matrix subtraction with incompatible dimensions.\n";
            return *this;
        }
        std::transform(
            begin(), end(),
            rhs.begin(),
            begin(),
            [] (const auto& a, const auto& b)
            {
                return a - b;
            }
        );
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, const Matrix& mat)
    {
        size_type j {0};
        for (size_type lidx {0}; lidx < mat.m_length; ++lidx)
        {
            j = mat.cidx(lidx);
            if (j != 0)
                out << "\t";
            out << mat.m_arr[lidx]; 
            if (j == (mat.m_cols - 1))
                out << "\n";
        }
        return out;
    }

private:
    // Data Members
    std::unique_ptr<double[]> m_arr {};
    size_type m_length { 0 };
    size_type m_rows {};
    size_type m_cols {};
};

inline bool incompatibleDims(const Matrix& a, const Matrix& b, bool mult)
{ 
    if (mult)
        return (a.cols() != b.rows());
    return (a.rows() != b.rows()) || (a.cols() != b.cols()); 
}

// Arithmetic Operators
// // Matrix Addition
inline Matrix operator+(const Matrix& lhs, const Matrix& rhs)
{
    if (incompatibleDims(lhs, rhs))
    {
        std::cerr << "Matrix addition with incompatible dimensions.\n";
        return Matrix{0,0};
    }
    Matrix out { rhs.rows(), rhs.cols() };
    std::transform(
        lhs.begin(), lhs.end(),
        rhs.begin(),
        out.begin(),
        [] (const auto& a, const auto& b)
        {
            return a + b;
        }
    );
    return out;
}
// // Matrix Subtraction
inline Matrix operator-(const Matrix& lhs, const Matrix& rhs)
{
    if (incompatibleDims(lhs, rhs))
    {
        std::cerr << "Matrix subtraction with incompatible dimensions.\n";
        return Matrix{0,0};
    }
    Matrix out { rhs.rows(), rhs.cols() };
    std::transform(
        lhs.begin(), lhs.end(),
        rhs.begin(),
        out.begin(),
        [] (const auto& a, const auto& b)
        {
            return a - b;
        }
    );
    return out;
}
// // Matrix Multiplication
inline Matrix operator*(const Matrix& lhs, const Matrix& rhs)
{
    if (incompatibleDims(lhs, rhs, true))
    {
        std::cerr << "Matrix multiplication with incompatible dimensions.\n";
        return Matrix{0,0};
    }
    Matrix out { lhs.rows(), rhs.cols() };

    // For each element in out:
    // // Compute i, j of the output matrix
    // // Dot lhs[i, :] with rhs[:, j]
    using size_type = Matrix::size_type;
    size_type ridx {0};
    size_type cidx {0};
    size_type length { out.length() };
    size_type inner { lhs.cols() };
    double sum { 0.0 };
    for (size_type l {0}; l < length; ++l)
    {
        ridx = out.ridx(l);
        cidx = out.cidx(l);
        sum = 0.0;
        for (size_type i {0}; i < inner; ++i)
        {
            sum += lhs[ridx, i] * rhs[i, cidx];
        }
        out[ridx, cidx] = sum;
    }
    return out;
}
// // Scalar Multiplication
inline Matrix operator*(double v, const Matrix& rhs)
{
    // Don't utilize the copy constructor to get matrix of same size (expensive)
    Matrix out { rhs.rows(), rhs.cols() };
    std::transform(
        rhs.begin(), rhs.end(),
        out.begin(),
        [v] (const auto& a)
        {
            return a*v;
        }
    );
    return out;
}
inline Matrix operator*(const Matrix& lhs, double v) { return operator*(v, lhs); }
// // Scalar Division
inline Matrix operator/(const Matrix& lhs, double v)
{
    if (v == 0.0)
    {
        std::cerr << "Division by zero not allowed.\n";
        return Matrix {0, 0};
    }
    return operator*(1.0/v, lhs);
}
// // Scalar Addition
inline Matrix operator+(double v, const Matrix& rhs)
{
    Matrix out { rhs.rows(), rhs.cols() };
    std::transform(
        rhs.begin(), rhs.end(),
        out.begin(),
        [v] (const auto& a)
        {
            return a+v;
        }
    );
    return out;
}
inline Matrix operator+(const Matrix& lhs, double v) { return operator+(v, lhs); }
// // Scalar Subtraction
inline Matrix operator-(double v, const Matrix& rhs) { return operator+(v, -rhs); }
inline Matrix operator-(const Matrix& lhs, double v) { return operator+(-v, lhs); }

} // end namespace linalg

#endif // MATRIX_H