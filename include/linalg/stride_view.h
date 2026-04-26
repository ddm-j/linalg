#ifndef STRIDE_VIEW_H
#define STRIDE_VIEW_H

#include <compare>

namespace linalg {

template <typename T>
class Matrix;

template <typename T>
class StrideIterator
{
public:
    using difference_type = std::ptrdiff_t;
    using value_type = T;

    StrideIterator() = default;

    StrideIterator(difference_type stride, T* ptr)
        : m_stride { stride }
        , m_ptr { ptr }
    {}

    // Arithmetic Operators
    T& operator*() const { return *m_ptr; };
    StrideIterator operator+(difference_type n) const { return StrideIterator(m_stride, m_ptr + n*m_stride); }
    StrideIterator operator-(difference_type n) const { return StrideIterator(m_stride, m_ptr - n*m_stride); }
    difference_type operator-(const StrideIterator<T>& other) const { return m_ptr - other.m_ptr; }
    StrideIterator& operator++() { m_ptr += m_stride; return *this; }
    StrideIterator operator++(int) { StrideIterator<T> temp{*this}; m_ptr += m_stride; return temp; }
    StrideIterator& operator--() { m_ptr -= m_stride; return *this; }
    StrideIterator operator--(int) { StrideIterator<T> temp{*this}; m_ptr -= m_stride; return temp; }

    // Access
    T& operator[](difference_type n) const { return *(m_ptr + n*m_stride); }
    difference_type getStride() const { return m_stride; }
    T* getPtr() const { return m_ptr; }

    // Comparison (weak ordering because different stride makes objects non sub)
    std::weak_ordering operator<=>(const StrideIterator<T>& other) const { return m_ptr <=> other.m_ptr; }
    bool operator==(const StrideIterator<T>& other) const { return m_ptr == other.m_ptr; }

    std::weak_ordering operator<=>(const T* other) const { return m_ptr <=> other; }
    bool operator==(const T* other) const { return m_ptr == other; }

private:
    difference_type m_stride {};
    T* m_ptr { nullptr };
};

template <typename T>
class StrideView
{
public:
    using size_type = StrideIterator<T>::difference_type;

    StrideView(T* ptr, size_type stride, size_type count)
        : m_ptr { ptr }
        , m_stride { stride }
        , m_count { count }
    {}

    StrideIterator<T> begin() const { return StrideIterator<T>(m_stride, m_ptr); }
    StrideIterator<T> end() const { return StrideIterator<T>(m_stride, m_ptr + m_stride*m_count); }

    // Access
    T* getPtr() const { return m_ptr; }
    size_type getStride() const { return m_stride; }
    size_type getCount() const { return m_count; }
    T& operator[](size_type n) const { return *(m_ptr + n*m_stride); }

private:
    T* m_ptr {nullptr};
    size_type m_stride {};
    size_type m_count {};
};

template <typename T>
class MatrixView
{
public:
    using size_type = linalg::Matrix<T>::size_type;

    // RAII
    MatrixView(T* ptr, size_type rows, size_type cols, size_type row_stride, size_type col_stride)
        : m_ptr { ptr }
        , m_rows { rows }
        , m_cols { cols }
        , m_row_stride { row_stride }
        , m_col_stride { col_stride }
    {}
    MatrixView(linalg::Matrix<T>& mat)
        : MatrixView(mat.begin(), mat.rows(), mat.cols(), mat.cols(), 1)
    {}

    // General
    size_type rows() const { return m_rows; }
    size_type cols() const { return m_cols; }
    size_type ridx(size_type lidx) const { return lidx / m_cols; }
    size_type cidx(size_type lidx) const { return lidx % m_cols; }
    size_type length() const { return m_rows*m_cols; }

    // Access Operators
    T& operator[](size_type i, size_type j) { return m_ptr[i * m_row_stride + j * m_col_stride]; }
    const T& operator[](size_type i, size_type j) const { return m_ptr[i * m_row_stride + j * m_col_stride]; }
    T& operator[](size_type l) { return operator[](ridx(l), cidx(l)); }
    const T& operator[](size_type l) const { return operator[](ridx(l), cidx(l)); }

private:
    T* m_ptr {nullptr};
    size_type m_rows {};
    size_type m_cols {};
    size_type m_row_stride {};
    size_type m_col_stride {};
};

}

#endif // STRIDE_VIEW_H