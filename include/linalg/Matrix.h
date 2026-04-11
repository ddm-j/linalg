#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <memory>
#include <iterator>
#include <initializer_list>
#include <iostream>

namespace linalg {

//==============================================================================
// FORWARD DECLARACTIONS
//==============================================================================
template <typename T>
class Matrix;

template <typename T>
bool incompatibleDims(const Matrix<T>& a, const Matrix<T>& b, bool mult = false);

template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator*(T v, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, T v);

template <typename T>
Matrix<T> operator/(const Matrix<T>& lhs, T v);

template <typename T>
Matrix<T> operator+(T v, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, T v);

template <typename T>
Matrix<T> operator-(T v, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, T v);

//==============================================================================
// CLASS DEFINITION
//==============================================================================
template <typename T>
class Matrix
{
public:
    using size_type = std::size_t;

    // RAII
    // // Constructor(s)
    Matrix(size_type rows, size_type cols);
    Matrix(std::initializer_list<T> list);
    Matrix(std::initializer_list<std::initializer_list<T>> list);
    // // Destructor
    ~Matrix();
    // // Copy Constructor
    Matrix(const Matrix<T>& mat);
    // // Copy Assignment Operator
    Matrix& operator=(const Matrix<T>& mat);
    // // Move Constructor
    Matrix(Matrix<T>&& mat) noexcept;
    // // Move Assignment Operator
    Matrix<T>& operator=(Matrix<T>&& mat) noexcept;

    // General Utility
    size_type rows() const { return m_rows; }
    size_type cols() const { return m_cols; }
    size_type ridx(size_type lidx) const { return lidx / m_cols; }
    size_type cidx(size_type lidx) const { return lidx % m_cols; }
    size_type length() const { return m_length; }

    // Member Functions
    const T& at(size_type i, size_type j) const { return m_arr[i * m_cols + j]; }
    T& at(size_type i, size_type j) { return m_arr[i * m_cols + j]; }
    T* begin() { return m_arr.get(); }
    T* end() { return begin() + m_length; }
    const T* begin() const { return m_arr.get(); }
    const T* end() const { return begin() + m_length; }

    // Operator Overloads
    T& operator[](size_type i, size_type j) { return m_arr[i * m_cols + j]; }
    const T& operator[](size_type i, size_type j) const { return m_arr[i * m_cols + j]; }
    T& operator[](size_type l) { return m_arr[l]; }
    const T& operator[](size_type l) const { return m_arr[l]; }
    Matrix<T> operator-() const { return operator*(-1.0, *this); }
    Matrix<T>& operator+=(const Matrix<T>& rhs);
    Matrix<T>& operator-=(const Matrix<T>& rhs);

private:
    // Data Members
    std::unique_ptr<T[]> m_arr {};
    size_type m_length { 0 };
    size_type m_rows {};
    size_type m_cols {};
};

//==============================================================================
// IMPLEMENTATIONS
//==============================================================================
// RAII
// // Constructor
template <typename T>
Matrix<T>::Matrix(size_type rows, size_type cols)
    : m_arr { std::make_unique<T[]>(rows * cols) }
    , m_length { rows*cols }
    , m_rows { rows }
    , m_cols { cols }
{}

// // Destructor
template <typename T>
Matrix<T>::~Matrix()
{}

// // Copy Constructor
template <typename T>
Matrix<T>::Matrix(const Matrix<T>& mat)
    : m_arr { std::make_unique<T[]>(mat.m_rows * mat.m_cols) }
    , m_length { mat.m_length }
    , m_rows { mat.m_rows }
    , m_cols { mat.m_cols }
{
    // std::cout << "Copy Constructor\n";
    std::copy_n(mat.begin(), mat.m_length, begin());
}

// // Copy Assignment Operator
template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& mat)
{
    // std::cout << "Copy Assignment\n";
    if (&mat == this)
        return *this;
    
    // Allow copy if length is the same
    if (m_length != mat.m_length)
        m_arr = std::make_unique<T[]>(mat.m_length);

    std::copy_n(mat.begin(), mat.m_length, begin());
    m_length = mat.m_length;
    m_rows = mat.m_rows;
    m_cols = mat.m_cols;
    return *this;
}

// // Move Constructor
template <typename T>
Matrix<T>::Matrix(Matrix<T>&& mat) noexcept
    : m_arr { std::move(mat.m_arr) }
    , m_length { mat.m_length }
    , m_rows { mat.m_rows }
    , m_cols { mat.m_cols }
{
    // std::cout << "Move Constructor\n";
    // Set Empty State
    mat.m_length = 0;
    mat.m_rows = 0;
    mat.m_cols = 0;
}

// // Move Assignment Operator
template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& mat) noexcept
{
    // std::cout << "Move Assignment\n";
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

// Additional Constructors
template <typename T>
Matrix<T>::Matrix(std::initializer_list<T> list)
    : Matrix<T>::Matrix(1, list.size())
{
    std::copy(list.begin(), list.end(), m_arr.get());
}

template <typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list)
    : Matrix<T>::Matrix(list.size(), list.begin()->size())
{
    using size_type = std::initializer_list<T>::size_type;
    for (size_type i {0}; i < list.size(); ++i)
    {
        std::copy_n((list.begin()+i)->begin(), m_cols, m_arr.get()+i*m_cols);
    }
}

// Member Operators
template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs)
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

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs)
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

// Utility
template <typename T>
bool incompatibleDims(const Matrix<T>& a, const Matrix<T>& b, bool mult)
{ 
    if (mult)
        return (a.cols() != b.rows());
    return (a.rows() != b.rows()) || (a.cols() != b.cols()); 
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& mat)
{
    using size_type = Matrix<T>::size_type;

    size_type j {0};
    size_type length { mat.length() };
    size_type cols { mat.cols() };
    for (size_type lidx {0}; lidx < length; ++lidx)
    {
        j = mat.cidx(lidx);
        if (j != 0)
            out << "\t";
        out << mat[lidx]; 
        if (j == (cols - 1))
            out << "\n";
    }
    return out;
}

// Arithmetic Operators
// // Matrix Addition
template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (incompatibleDims(lhs, rhs))
    {
        std::cerr << "Matrix addition with incompatible dimensions.\n";
        return Matrix<T>{0, 0};
    }
    Matrix<T> out { rhs.rows(), rhs.cols() };
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
template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (incompatibleDims(lhs, rhs))
    {
        std::cerr << "Matrix subtraction with incompatible dimensions.\n";
        return Matrix<T>{0, 0};
    }
    Matrix<T> out { rhs.rows(), rhs.cols() };
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
template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (incompatibleDims(lhs, rhs, true))
    {
        std::cerr << "Matrix multiplication with incompatible dimensions.\n";
        return Matrix<T>{0, 0};
    }
    Matrix<T> out { lhs.rows(), rhs.cols() };

    // For each element in out:
    // // Compute i, j of the output matrix
    // // Dot lhs[i, :] with rhs[:, j]
    using size_type = Matrix<T>::size_type;
    size_type ridx {0};
    size_type cidx {0};
    size_type length { out.length() };
    size_type inner { lhs.cols() };
    T sum { }; // Should zero initialize
    for (size_type l {0}; l < length; ++l)
    {
        ridx = out.ridx(l);
        cidx = out.cidx(l);
        sum = T{};
        for (size_type i {0}; i < inner; ++i)
        {
            sum += lhs[ridx, i] * rhs[i, cidx];
        }
        out[ridx, cidx] = sum;
    }
    return out;
}

// // Scalar Multiplication
template <typename T>
Matrix<T> operator*(T v, const Matrix<T>& rhs)
{
    // Don't utilize the copy constructor to get matrix of same size (expensive)
    Matrix<T> out { rhs.rows(), rhs.cols() };
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
template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, T v) { return operator*(v, lhs); }

// // Scalar Division
template <typename T>
Matrix<T> operator/(const Matrix<T>& lhs, T v)
{
    if (v == T{})
    {
        std::cerr << "Division by zero not allowed.\n";
        return Matrix<T> {0, 0};
    }
    return operator*(1.0/v, lhs);
}

// // Scalar Addition
template <typename T>
Matrix<T> operator+(T v, const Matrix<T>& rhs)
{
    Matrix<T> out { rhs.rows(), rhs.cols() };
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
template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, T v) { return operator+(v, lhs); }

// // Scalar Subtraction
template <typename T>
Matrix<T> operator-(T v, const Matrix<T>& rhs) { return operator+(v, -rhs); }
template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, T v) { return operator+(-v, lhs); }

} // end namespace linalg

#endif // MATRIX_H