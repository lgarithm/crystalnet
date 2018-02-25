#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include <misaka.h>
#include <misaka/core/debug.hpp> // for LOG_TENSOR_USAGE
#include <misaka/core/idx.hpp>   // for shape_t
#include <misaka/core/shape.hpp> // for shape_t

struct tensor_t;
struct tensor_ref_t;

struct tensor_t {
    const shape_t shape;
    const uint8_t dtype;
    const std::unique_ptr<uint8_t[]> _data;
    void *data;

    explicit tensor_t(const shape_t &shape,
                      uint8_t dtype = idx_type<float>::type)
        : shape(shape), dtype(dtype),
          _data(new uint8_t[dtype_size(dtype) * shape.dim()]), data(_data.get())
    {
        LOG_TENSOR_USAGE(shape, dtype_size(dtype));
        memset(data, 0, dtype_size(dtype) * shape.dim());
    }

    // TODO: support initializers

    tensor_ref_t operator[](uint32_t idx) const;
};

struct tensor_ref_t {
    const shape_t shape;
    const uint8_t dtype;
    void *data;

    tensor_ref_t(const shape_t &shape, uint8_t dtype, void *data)
        : shape(shape), dtype(dtype), data(data)
    {
    }

    explicit tensor_ref_t(const tensor_t &tensor)
        : shape(tensor.shape), dtype(tensor.dtype), data(tensor.data)
    {
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
    R *data;

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
        R s = 0;
        auto n = shape.dim();
        for (auto i = 0; i < n; ++i) {
            s += data[i];
        }
        return s / n;
    }
};

template <typename R, typename T>
tensor_t *cast_to(const r_tensor_ref_t<T> &tensor)
{
    auto t = new tensor_t(tensor.shape, idx_type<R>::type);
    auto r = r_tensor_ref_t<R>(*t);
    uint32_t n = t->shape.dim();
    for (auto i = 0; i < n; ++i) {
        // TODO: use static_cast
        r.data[i] = R(tensor.data[i]);
    }
    return t;
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