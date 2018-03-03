#pragma once
#include <cassert>
#include <cstdint>

template <typename T> struct matrix_ref_t {
    const uint32_t m;
    const uint32_t n;
    T *const data;

    matrix_ref_t(uint32_t m, uint32_t n, T *data) : m(m), n(n), data(data) {}

    inline T &operator()(uint32_t i, uint32_t j) const
    {
        return data[i * n + j];
    }
};

template <typename T> struct vector_ref_t {
    const uint32_t n;
    T *const data;

    vector_ref_t(uint32_t n, T *data) : n(n), data(data) {}

    inline T &operator()(uint32_t i) const { return data[i]; }
};

template <typename T> T equally(T m, T n)
{
    assert(m == n);
    return m;
}

template <typename T> T equally(T k, T m, T n)
{
    assert(k == m);
    assert(m == n);
    return k;
}

#include <crystalnet/core/tensor.hpp>

template <typename T> auto as_vector_ref(const tensor_ref_t &tensor)
{
    assert(tensor.dtype == idx_type<T>::type);
    assert(tensor.shape.rank() == 1);
    return vector_ref_t<T>(tensor.shape.dims[0], (T *)tensor.data);
}

template <typename T> auto as_matrix_ref(const tensor_ref_t &tensor)
{
    assert(tensor.dtype == idx_type<T>::type);
    assert(tensor.shape.rank() == 2);
    auto m = tensor.shape.dims[0];
    auto n = tensor.shape.dims[1];
    return matrix_ref_t<T>(m, n, (T *)tensor.data);
}

template <typename T> auto as_col_matrix_ref(const vector_ref_t<T> &vector)
{
    return matrix_ref_t<T>(vector.n, 1, vector.data);
}

template <typename T> auto as_row_matrix_ref(const vector_ref_t<T> &vector)
{
    return matrix_ref_t<T>(1, vector.n, vector.data);
}

template <typename T> uint32_t argmax(const vector_ref_t<T> &vector)
{
    return std::max_element(vector.data, vector.data + vector.n) - vector.data;
}