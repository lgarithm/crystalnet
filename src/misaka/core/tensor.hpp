#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <misaka.h>
#include <misaka/core/debug.hpp> // for LOG_TENSOR_USAGE
#include <misaka/core/idx.hpp>
#include <misaka/core/shape.hpp> // for shape_t

struct tensor_t {
    const shape_t shape;
    const uint8_t dtype;
    const std::unique_ptr<uint8_t[]> _data;
    void *const data;

    explicit tensor_t(const shape_t &shape,
                      uint8_t dtype = idx_type<float>::type)
        : shape(shape), dtype(dtype),
          _data(new uint8_t[dtype_size(dtype) * shape.dim()]), data(_data.get())
    {
        LOG_TENSOR_USAGE(shape, dtype_size(dtype));
        memset(data, 0, dtype_size(dtype) * shape.dim());
    }

    // TODO: support initializers
};

struct tensor_ref_t {
    const shape_t shape;
    const uint8_t dtype;
    void *const data;

    tensor_ref_t(const shape_t &shape, uint8_t dtype, void *data)
        : shape(shape), dtype(dtype), data(data)
    {
    }

    explicit tensor_ref_t(const tensor_t &tensor)
        : shape(tensor.shape), dtype(tensor.dtype), data(tensor.data)
    {
    }

    tensor_ref_t operator[](uint32_t idx) const
    {
        assert(idx < shape.len());
        if (shape.rank() == 0) {
            return *this;
        }
        shape_t new_shape(
            std::vector<uint32_t>(shape.dims.begin() + 1, shape.dims.end()));
        uint32_t offset = idx * new_shape.dim() * dtype_size(dtype);
        return tensor_ref_t(new_shape, dtype, (uint8_t *)(data) + offset);
    }
};

struct tensor_ref_list_t {
    const std::vector<tensor_ref_t> _args;
    explicit tensor_ref_list_t(const std::vector<tensor_ref_t> &args)
        : _args(args)
    {
    }
    uint8_t arity() const { return _args.size(); }
    tensor_ref_t operator[](uint8_t i) const { return _args[i]; }
};

inline tensor_ref_t ref(const tensor_t &tensor) { return tensor_ref_t(tensor); }

template <typename R> struct r_tensor_ref_t {
    const shape_t shape;
    R *const data;

    explicit r_tensor_ref_t(const tensor_t &t)
        : shape(t.shape), data((R *)t.data)
    {
        assert(idx_type<R>::type == t.dtype);
    }

    explicit r_tensor_ref_t(const tensor_ref_t &t)
        : shape(t.shape), data((R *)t.data)
    {
        assert(idx_type<R>::type == t.dtype);
    }

    R max() const { return *std::max_element(data, data + shape.dim()); }
    R min() const { return *std::min_element(data, data + shape.dim()); }
    R mean() const
    {
        const auto n = shape.dim();
        return std::accumulate(data, data + n, (R)0) / n;
    }
    void fill(R x) const { std::fill(data, data + shape.dim(), x); }
    void copy(const r_tensor_ref_t<R> &r)
    {
        const auto n = shape.dim();
        assert(n == r.shape.dim());
        std::memcpy(data, r.data, n * sizeof(R));
    }
};

template <typename R, typename T> tensor_t *cast_to(const r_tensor_ref_t<T> &t)
{
    auto r = new tensor_t(t.shape, idx_type<R>::type);
    std::transform(t.data, t.data + t.shape.dim(), r_tensor_ref_t<R>(*r).data,
                   [](T x) { return (R)x; });
    return r;
}

namespace std
{
inline string to_string(const tensor_t &t)
{
    return string("tensor(dtype=") + dtype_name(t.dtype) + ",rank" +
           to_string(t.shape.rank()) + ",dim" + to_string(t.shape.dim());
}
inline string to_string(const tensor_ref_t &t)
{
    return string("tensor_ref(dtype=") + dtype_name(t.dtype) +
           ",rank=" + to_string(t.shape.rank()) +
           ",dim=" + to_string(t.shape.dim()) + ",shape=" + to_string(t.shape) +
           ")";
}
}

template <typename T> void print(const r_tensor_ref_t<T> &r)
{
    constexpr const char *const fmt = "min: %12f    meam: %12f    max: %12f\n";
    printf(fmt, r.min(), r.mean(), r.max());
}
