#ifndef LINALG_MATRIX_H
#define LINALG_MATRIX_H

#include <cstddef>
#include <memory>
#include <iterator>
#include <initializer_list>
#include <exception>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <ranges>
#include <algorithm>
#include <linalg/stride_view.h>

namespace linalg {

//==============================================================================
// OSTREAM GUARD
//==============================================================================
class IosFlagSaver {
public:
    explicit IosFlagSaver(std::ostream& os) 
        : os(os)
        , flags(os.flags()) 
        , precision(os.precision())
    {}
    ~IosFlagSaver() { os.flags(flags); os.precision(precision); }
    
    // Prevent copying to avoid multiple restorations
    IosFlagSaver(const IosFlagSaver&) = delete;
    IosFlagSaver& operator=(const IosFlagSaver&) = delete;

private:
    std::ostream& os;
    std::ios_base::fmtflags flags;
    std::streamsize precision; 
};

//==============================================================================
// FORWARD DECLARACTIONS
//==============================================================================
template <typename T>
class Matrix;

template <typename T>
bool differentDims(const Matrix<T>& a, const Matrix<T>& b);

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
    Matrix(size_type rows, size_type cols, std::initializer_list<T> list);
    template <typename U>
    explicit Matrix(const MatrixView<U>& view); // Allow conversion from view
    // // Destructor
    ~Matrix() = default;
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
    MatrixView<T> transpose() { return MatrixView<T>(begin(), m_cols, m_rows, 1, m_cols); }
    MatrixView<const T> transpose() const { return MatrixView<const T>(begin(), m_cols, m_rows, 1, m_cols); }

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
    Matrix<T> operator-() const { return operator*(T{-1}, *this); }
    Matrix<T>& operator+=(const Matrix<T>& rhs);
    Matrix<T>& operator-=(const Matrix<T>& rhs);

    // Row View
    StrideView<T> rowit(size_type i) { return StrideView<T>(m_arr.get() + i*m_cols, 1, static_cast<ptr_diff>(m_cols)); }
    StrideView<const T> rowit(size_type i) const { return StrideView<const T>(m_arr.get() + i*m_cols, 1, static_cast<ptr_diff>(m_cols)); }

    // Column View
    StrideView<T> colit(size_type j) { return StrideView<T>(m_arr.get() + j, static_cast<ptr_diff>(m_cols), static_cast<ptr_diff>(m_rows)); }
    StrideView<const T> colit(size_type j) const { return StrideView<const T>(m_arr.get() + j, static_cast<ptr_diff>(m_cols), static_cast<ptr_diff>(m_rows)); }

    // Diagonal View
    StrideView<T> diagit() { return StrideView<T>(m_arr.get(), static_cast<ptr_diff>(m_cols + 1), static_cast<ptr_diff>(std::min(m_rows, m_cols))); }
    StrideView<const T> diagit() const { return StrideView<const T>(m_arr.get(), static_cast<ptr_diff>(m_cols + 1), static_cast<ptr_diff>(std::min(m_rows, m_cols))); }

    // Hidden Friends
    // // Equality
    friend bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        if (lhs.rows() != rhs.rows())
            return false;
        if (lhs.cols() != rhs.cols())
            return false;

        for (auto&& [a, b] : std::views::zip(lhs, rhs))
        {
            if (a != b)
                return false;
        }
        return true;
    }
    // // Inequality
    friend bool operator!=(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        return !operator==(lhs, rhs);
    }

    // // Matrix Addition
    friend Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        if (differentDims(lhs, rhs))
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
    friend Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        if (differentDims(lhs, rhs))
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

    // Matrix Multiplication (standard)
#ifndef BLOCKED_MATMUL_ENABLED
    friend Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        if (lhs.cols() != rhs.rows())
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
#else 
    // Matrix Multiplication (Blocked)
    friend Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        if (lhs.cols() != rhs.rows())
        {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication.");
        }
        Matrix<T> out(lhs.rows(), rhs.cols());

        static constexpr size_type b = 64;
        size_type M = lhs.rows();
        size_type K = lhs.cols();   // == rhs.rows(), checked by incompatibleDims
        size_type N = rhs.cols();

        for (size_type i0 {0}; i0 < M; i0 += b)
        {
            for (size_type j0 {0}; j0 < N; j0 += b)
            {
                for (size_type k0 {0}; k0 < K; k0 += b)
                {
                    for (size_type i {i0}; i < std::min(i0+b, M); ++i)
                    {
                        for (size_type k {k0}; k < std::min(k0+b, K); ++k)
                        {
                            for (size_type j {j0}; j < std::min(j0+b, N); ++j)
                            {
                                out[i, j] += lhs[i, k] * rhs[k, j];
                            }
                        }
                    }
                }
            }
        }
        return out;
    }
#endif
    // // Scalar Multiplication
    friend Matrix<T> operator*(T v, const Matrix<T>& rhs)
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
    friend Matrix<T> operator*(const Matrix<T>& lhs, T v) { return operator*(v, lhs); }

    // // Scalar Division
    friend Matrix<T> operator/(const Matrix<T>& lhs, T v)
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
    friend Matrix<T> operator+(T v, const Matrix<T>& rhs)
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
    friend Matrix<T> operator+(const Matrix<T>& lhs, T v) { return operator+(v, lhs); }

    // // Scalar Subtraction
    friend Matrix<T> operator-(T v, const Matrix<T>& rhs) { return operator+(v, -rhs); }
    friend Matrix<T> operator-(const Matrix<T>& lhs, T v) { return operator+(-v, lhs); }

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
    // Check for Empty List
    if (!list.begin())
    {
        throw std::invalid_argument("initializer list is empty.");
    }

    std::copy(list.begin(), list.end(), m_arr.get());
}

template <typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list)
    : Matrix<T>::Matrix(list.size(), list.begin()->size())
{
    // Check for Empty List
    if (!list.begin())
    {
        throw std::invalid_argument("initializer list is empty.");
    }

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

template <typename T>
Matrix<T>::Matrix(size_type rows, size_type cols, std::initializer_list<T> list)
    : Matrix<T>::Matrix(rows, cols)
{
    if (rows*cols != list.size())
        throw std::invalid_argument("initializer list size does not match rows*cols");

    std::copy(list.begin(), list.end(), m_arr.get());
}

template <typename T>
template <typename U>
Matrix<T>::Matrix(const MatrixView<U>& view)
    : Matrix<T>::Matrix(view.rows(), view.cols())
{
    for (size_type i {0}; i < m_length; ++i)
    {
        m_arr[i] = view[i];
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
    if (differentDims(*this, rhs))
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
    if (differentDims(*this, rhs))
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
bool differentDims(const Matrix<T>& a, const Matrix<T>& b)
{ 
    return (a.rows() != b.rows()) || (a.cols() != b.cols()); 
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& mat)
{
    IosFlagSaver guard(out);

    using size_type = Matrix<T>::size_type;
    out << std::fixed << std::setprecision(3);

    size_type length { mat.length() };
    size_type cols { mat.cols() };

    // Find Max Width Column
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    int width { std::accumulate(
        mat.begin(),
        mat.end(),
        0,
        [&oss](int w, const auto& v)
        {
            oss.str("");
            oss << v;
            if (int a {static_cast<int>(oss.str().size())}; a > w)
                w = a;
            return w;
        }
    ) };
    width += 2;

    size_type j {0};
    for (size_type lidx {0}; lidx < length; ++lidx)
    {
        j = mat.cidx(lidx);
        out << std::setw(width) << mat[lidx]; 
        if (j == (cols - 1))
            out << "\n";
    }

    return out;
}

//==============================================================================
// NON MEMBER / NON FRIEND UTILITIES
//==============================================================================
template <typename T>
bool is_lowertri(const Matrix<T>& A, int fac = 100)
{
    using size_type = Matrix<T>::size_type;
    size_type rows { A.rows() };
    size_type cols { A.cols() };
    T eps { fac * std::numeric_limits<T>::epsilon() };
    for (size_type i {0}; i < std::min(rows, cols); ++i)
    {
        for (size_type j {i + 1}; j < cols; ++j)
        {
            if (std::abs(A[i, j]) > eps)
                return false;
        }
    }
    return true;
}

template <typename T>
bool is_uppertri(const Matrix<T>& A, int fac = 100)
{
    using size_type = Matrix<T>::size_type;
    size_type rows { A.rows() };
    size_type cols { A.cols() };
    T eps { fac * std::numeric_limits<T>::epsilon() };
    for (size_type i {0}; i < std::min(rows, cols); ++i)
    {
        for (size_type j {0}; j < i; ++j)
        {
            if (std::abs(A[i, j]) > eps)
                return false;
        }
    }
    return true;
}

} // end namespace linalg

#endif // LINALG_MATRIX_H