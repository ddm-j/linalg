#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <memory>
#include <iterator>
#include <initializer_list>
#include <exception>
#include <iostream>
#include <linalg/stride_view.h>

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
    using ptr_diff = StrideView<T>::size_type;

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

    // Statics
    static Matrix<T> ones(size_type rows, size_type cols);
    static Matrix<T> eye(size_type rows, size_type cols);

    // General Utility
    size_type rows() const { return m_rows; }
    size_type cols() const { return m_cols; }
    size_type ridx(size_type lidx) const { return lidx / m_cols; }
    size_type cidx(size_type lidx) const { return lidx % m_cols; }
    size_type length() const { return m_length; }

    // Member Functions
    T* begin() { return m_arr.get(); }
    T* end() { return begin() + m_length; }
    const T* begin() const { return m_arr.get(); }
    const T* end() const { return begin() + m_length; }

    // Operator Overloads
    T& operator[](size_type i, size_type j) { return m_arr[i * m_cols + j]; }
    const T& operator[](size_type i, size_type j) const { return m_arr[i * m_cols + j]; }
    T& operator[](size_type l) { return m_arr[l]; }
    const T& operator[](size_type l) const { return m_arr[l]; }
    Matrix<T> operator-() const { return operator*(-1, *this); }
    Matrix<T>& operator+=(const Matrix<T>& rhs);
    Matrix<T>& operator-=(const Matrix<T>& rhs);

    // Row View
    StrideView<T> rowit(size_type i) { return StrideView<T>(m_arr.get() + i*m_cols, 1, m_cols); }
    StrideView<const T> rowit(size_type i) const { return StrideView<const T>(m_arr.get() + i*m_cols, 1, m_cols); }

    // Column View
    StrideView<T> colit(size_type j) { return StrideView<T>(m_arr.get() + j, m_cols, m_rows); }
    StrideView<const T> colit(size_type j) const { return StrideView<const T>(m_arr.get() + j, m_cols, m_rows); }

    // Diagonal View
    StrideView<T> diagit() { return StrideView<T>(m_arr.get(), m_cols + 1, std::min(m_rows, m_cols)); }
    StrideView<const T> diagit() const { return StrideView<const T>(m_arr.get(), m_cols + 1, std::min(m_rows, m_cols)); }

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
    std::copy_n(mat.begin(), mat.m_length, begin());
}

// // Copy Assignment Operator
template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& mat)
{
    if (&mat == this)
        return *this;
    
    // Only re-allocate if length is not the same
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
    // Set Empty State
    mat.m_length = 0;
    mat.m_rows = 0;
    mat.m_cols = 0;
}

// // Move Assignment Operator
template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& mat) noexcept
{
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
    // Check for a jagged initializer
    auto expected = list.begin()->size();
    if (
        !std::all_of(list.begin(), list.end(),
        [expected](const auto& l)
        {
            return expected == l.size();
        })
    )
        throw std::invalid_argument("initializer lists are jagged (not same length)");

    using size_type = std::initializer_list<T>::size_type;
    for (size_type i {0}; i < list.size(); ++i)
    {
        std::copy_n((list.begin()+i)->begin(), m_cols, m_arr.get()+i*m_cols);
    }
}

// Static Functions
template <typename T>
Matrix<T> Matrix<T>::ones(size_type rows, size_type cols)
{
    Matrix<T> out(rows, cols);
    std::fill(out.begin(), out.end(), T(1));
    return out;
}

template <typename T>
Matrix<T> Matrix<T>::eye(size_type rows, size_type cols)
{
    Matrix<T> out(rows, cols);
    auto diag { out.diagit() };
    std::fill(diag.begin(), diag.end(), T(1));
    return out;
}

// Member Operators
template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs)
{
    if (incompatibleDims(*this, rhs))
    {
        throw std::invalid_argument("Matrix dimensions incompatible for addition.");
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
        throw std::invalid_argument("Matrix dimensions incompatible for subtraction.");
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
        throw std::invalid_argument("Matrix dimensions incompatible for addition.");
    }
    Matrix<T> out(rhs.rows(), rhs.cols());
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
        throw std::invalid_argument("Matrix dimensions incompatible for subtraction.");
    }
    Matrix<T> out(rhs.rows(), rhs.cols());
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
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication.");
    }
    Matrix<T> out(lhs.rows(), rhs.cols());

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
    Matrix<T> out(rhs.rows(), rhs.cols());
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
    if (v == T())
    {
        throw std::invalid_argument("Division by zero not allowed.");
    }
    Matrix<T> out(lhs.rows(), lhs.cols());
    std::transform(
        lhs.begin(), lhs.end(),
        out.begin(),
        [v] (const auto& a)
        {
            return a/v;
        }
    );
    return out;
}

// // Scalar Addition
template <typename T>
Matrix<T> operator+(T v, const Matrix<T>& rhs)
{
    Matrix<T> out(rhs.rows(), rhs.cols());
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