#pragma once

// #define ENABLE_TRACE_TENSOR_USAGE

#include <algorithm>   // for transform
#include <cassert>     // for assert
#include <cstddef>     // for size_t
#include <cstdint>     // for uint8_t
#include <cstring>     // for memcpy, memset
#include <functional>  // for function, plus
#include <memory>      // for unique_ptr
#include <type_traits> // for enable_if

#include "teavana/core/shape.hpp" // for dim, shape_t, sub, assert_eq_dim
#include "teavana/tracer.hpp"     // for tracer

namespace tea
{
template <typename R, uint8_t r> struct tensor;
template <typename R, uint8_t r> struct tensor_iterator;
template <typename R, uint8_t r> struct tensor_ref;

template <typename R> struct tensor_iterator<R, 0> {
    using elem_t = tensor_ref<R, 0>;
    R *pos;

    tensor_iterator(const shape_t<0> &, R *pos) : pos(pos) {}

    bool operator!=(const tensor_iterator &it) const { return pos != it.pos; }

    void operator++() { ++pos; }

    void _advance(size_t k) { pos += k; }

    elem_t operator*() const { return elem_t(shape(), pos); }
};

template <typename R, uint8_t r> struct tensor_iterator {
    using elem_t = tensor_ref<R, r>;
    const shape_t<r> shape;
    const size_t step;
    R *pos;

    tensor_iterator(const shape_t<r> &s, R *pos)
        : shape(s), step(dim(s)), pos(pos)
    {
    }

    bool operator!=(const tensor_iterator &it) const { return pos != it.pos; }

    void operator++() { pos += step; }

    void _advance(size_t k) { pos += k * step; }

    elem_t operator*() const { return elem_t(shape, pos); }
};

template <typename R> struct tensor_ref<R, 0> {
    static constexpr uint8_t rank = 0;
    using base_t = R;
    using own_t = tensor<R, 0>;

    const shape_t<0> shape;
    R *const data;

    tensor_ref(const own_t &t) : shape(t.shape), data(t.data) {}

    explicit tensor_ref(const shape_t<0> &s, base_t *data)
        : shape(s), data(data)
    {
    }

    base_t &operator[](size_t i) const { return data[i]; }
};

template <typename R, uint8_t r> struct tensor_ref {
    static constexpr uint8_t rank = r;
    using base_t = R;
    using own_t = tensor<R, r>;
    using iter_t = tensor_iterator<R, r - 1>;

    const shape_t<r> shape;
    R *const data;

    tensor_ref(const own_t &t) : shape(t.shape), data(t.data) {}

    explicit tensor_ref(const shape_t<r> &s, base_t *data)
        : shape(s), data(data)
    {
    }

    iter_t begin() const { return iter_t(sub(shape), data); }

    iter_t end() const { return iter_t(sub(shape), data + dim(shape)); }

    typename iter_t::elem_t operator[](size_t i) const
    {
        auto it = begin();
        it._advance(i);
        return *it;
    }

    template <typename... Args,
              typename = typename ::std::enable_if<sizeof...(Args) == r>::type>
    base_t &at(Args... args) const
    {
        return data[shape.idx(args...)];
    }
};

template <typename R> struct tensor<R, 0> {
    static constexpr uint8_t rank = 0;
    using base_t = R;
    using ref_t = tensor_ref<R, 0>;

    const shape_t<0> shape;
    ::std::unique_ptr<R[]> _data;
    R *data;

    explicit tensor(const shape_t<0> &s)
        : shape(s), _data(new R[dim(s)]), data(_data.get())
    {
        TRACE_TENSOR_USAGE(R, s);
        annihilate(*this);
    }
};

template <typename R, uint8_t r> struct tensor {
    static constexpr uint8_t rank = r;
    using base_t = R;
    using ref_t = tensor_ref<R, r>;
    using iter_t = tensor_iterator<R, r - 1>;

    const shape_t<r> shape;
    ::std::unique_ptr<R[]> _data;
    R *data;

    explicit tensor(const shape_t<r> &s)
        : shape(s), _data(new R[dim(s)]), data(_data.get())
    {
        TRACE_TENSOR_USAGE(R, s);
        annihilate(*this);
    }

    iter_t begin() const { return ref(*this).begin(); }

    iter_t end() const { return ref(*this).end(); }

    template <typename... Args,
              typename = typename ::std::enable_if<sizeof...(Args) == r>::type>
    base_t &at(Args... args) const
    {
        return data[shape.idx(args...)];
    }
};

template <typename R, uint8_t r> auto make_tensor(const shape_t<r> &s)
{
    return tensor<R, r>(s);
}

template <template <typename, uint8_t> class T, typename R, uint8_t r,
          typename = typename ::std::enable_if<(r > 0)>::type>
size_t len(const T<R, r> &t)
{
    return len(t.shape);
}

template <uint8_t k, template <typename, uint8_t> class T, typename R,
          uint8_t r>
size_t dim(const T<R, r> &t)
{
    return ::std::get<k>(t.shape.dims);
}

template <template <typename, uint8_t> class T, typename R, uint8_t r>
size_t data_size(const T<R, r> &t)
{
    return sizeof(R) * dim(t.shape);
}

template <template <typename, uint8_t> class T, typename R, uint8_t r>
void annihilate(const T<R, r> &t)
{
    memset(t.data, 0, data_size(t));
}

template <template <typename, uint8_t> class T1,
          template <typename, uint8_t> class T2, typename R, uint8_t r>
void operator+=(const T1<R, r> &a, const T2<R, r> &b)
{
    assert(a.shape == b.shape);
    ::std::transform(a.data, a.data + dim(a.shape), b.data, a.data,
                     ::std::plus<R>());
}

template <template <typename, uint8_t> class T, typename R>
R &scalar(const T<R, 0> &t)
{
    return t.data[0];
}

template <template <typename, uint8_t> class T, typename R, uint8_t r>
tensor_ref<R, r> ref(const T<R, r> &t)
{
    return tensor_ref<R, r>(t.shape, t.data);
}

template <template <typename, uint8_t> class T, typename R, uint8_t _,
          uint8_t r>
tensor_ref<R, r> ref_as(const T<R, _> &t, const shape_t<r> &shape)
{
    assert_eq_dim(t.shape, shape, __func__);
    return tensor_ref<R, r>(shape, t.data);
}

template <template <typename, uint8_t> class T, typename R, uint8_t r>
void fill(const T<R, r> &t, R x)
{
    const size_t n = dim(t.shape);
    for (size_t i = 0; i < n; ++i) {
        t.data[i] = x;
    }
}

template <template <typename, uint8_t> class T1,
          template <typename, uint8_t> class T2, typename R, uint8_t r>
void assign(const T1<R, r> &x, const T2<R, r> &y)
{
    assert(x.shape == y.shape);
    memcpy(x.data, y.data, data_size(y));
}

template <typename T1, typename T2>
void assign_with(
    const T1 &y, const T2 &x,
    const ::std::function<typename T1::base_t(typename T2::base_t)> &f)
{
    assert(x.shape == y.shape);
    ::std::transform(x.data, x.data + dim(x.shape), y.data, f);
}

template <typename R1, typename R2, uint8_t r>
tensor<R2, r> transform_tensor(const ::std::function<R2(R1)> &f,
                               const tensor<R1, r> &t)
{
    tensor<R2, r> s(t.shape);
    ::std::transform(t.data, t.data + dim(t.shape), s.data, f);
    return s;
}

template <template <typename, uint8_t> class T, typename R, uint8_t r>
auto slice(const T<R, r> &t, size_t i, size_t j)
{
    const auto s = sub(t.shape);
    return tensor_ref<R, r>(shape(j - i) * s, t.data + i * dim(s));
}

template <template <typename, uint8_t> class T, typename R, uint8_t r>
auto chunk(const T<R, r> &t, size_t k)
{
    const size_t l = len(t);
    const size_t n = l / k;
    // drop last l - n * k elements
    // timer::logf("chunk drops %d", l - n * k);
    return ref_as(slice(t, 0, n * k), shape(n, k) * sub(t.shape));
}
}
