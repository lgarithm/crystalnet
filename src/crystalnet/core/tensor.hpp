#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <crystalnet.h>
#include <crystalnet/core/error.hpp>
#include <crystalnet/core/idx.hpp>
#include <crystalnet/core/shape.hpp>

struct _tensor_meta_t {
    const uint8_t dtype;
    const shape_t shape;
    _tensor_meta_t(uint8_t dtype, const shape_t &shape)
        : dtype(dtype), shape(shape)
    {
    }
};

struct tensor_ref_t : _tensor_meta_t {
    void *const data;

    tensor_ref_t(const shape_t &shape, uint8_t dtype, void *data)
        : _tensor_meta_t(dtype, shape), data(data)
    {
    }

    tensor_ref_t operator[](uint32_t idx) const
    {
        check(idx < shape.len());
        if (shape.rank() == 0) {
            return *this;
        }
        const auto new_shape = shape.sub();
        const uint32_t offset = idx * new_shape.dim() * dtype_size(dtype);
        return tensor_ref_t(new_shape, dtype, (uint8_t *)(data) + offset);
    }

    tensor_ref_t slice(uint32_t i, uint32_t j) const
    {
        const auto sub_shape = shape.sub();
        const uint32_t offset = i * sub_shape.dim() * dtype_size(dtype);
        return tensor_ref_t(sub_shape.batch(j - i), dtype,
                            (uint8_t *)(data) + offset);
    }

    void copy_from(const tensor_ref_t &r) const
    {
        check(dtype == r.dtype);
        check(shape == r.shape);
        std::memcpy(data, r.data, shape.dim() * dtype_size(dtype));
    }
};

struct tensor_t : _tensor_meta_t {
    const std::unique_ptr<uint8_t[]> _data;
    void *const data;
    const tensor_ref_t self;

    explicit tensor_t(const shape_t &shape,
                      uint8_t dtype = idx_type<float>::type);
};

tensor_ref_t ref(const tensor_t &);

struct tensor_ref_list_t {
    const std::vector<tensor_ref_t> _args;
    explicit tensor_ref_list_t(const std::vector<tensor_ref_t> &args)
        : _args(args)
    {
    }
    uint8_t arity() const { return _args.size(); }
    tensor_ref_t operator[](uint32_t i) const { return _args[i]; }
    shape_list_t shapes() const
    {
        std::vector<shape_t> shapes;
        for (auto t : _args) {
            shapes.push_back(t.shape);
        }
        return shape_list_t(shapes);
    }
};

template <typename R> struct r_tensor_ref_t {
    const shape_t shape;
    R *const data;

    explicit r_tensor_ref_t(const tensor_t &t)
        : shape(t.shape), data((R *)t.data)
    {
        check(idx_type<R>::type == t.dtype);
    }

    explicit r_tensor_ref_t(const tensor_ref_t &t)
        : shape(t.shape), data((R *)t.data)
    {
        check(idx_type<R>::type == t.dtype);
    }

    R max() const { return *std::max_element(data, data + shape.dim()); }
    R min() const { return *std::min_element(data, data + shape.dim()); }
    R mean() const
    {
        const auto n = shape.dim();
        return std::accumulate(data, data + n, (R)0) / n;
    }
    void fill(R x) const { std::fill(data, data + shape.dim(), x); }
    void fill_uni() const { fill(1.0 / shape.dim()); }
    void copy(const r_tensor_ref_t<R> &r)
    {
        const auto n = shape.dim();
        check(n == r.shape.dim());
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

template <typename R, uint8_t r> struct ranked_tensor_ref_t {
    ranked_shape_t<r> shape;
    R *const data;
    ranked_tensor_ref_t(const ranked_shape_t<r> &shape, R *data)
        : shape(shape), data(data)
    {
    }

    void fill(R x) const { std::fill(data, data + shape.dim(), x); }

    auto operator[](uint32_t idx) const
    {
        static_assert(r > 0);
        check(idx < shape.dims[0]);
        const shape_t _new_shape(
            std::vector<uint32_t>(shape.dims.begin() + 1, shape.dims.end()));
        const auto new_shape = ranked<r - 1>(_new_shape);
        return ranked_tensor_ref_t<R, r - 1>(new_shape,
                                             data + idx * new_shape.dim());
    }

    template <typename... I> R &at(I... i) const
    {
        return data[shape.idx(i...)];
    }
};

template <uint8_t r, typename T>
ranked_tensor_ref_t<T, r> ranked(const tensor_ref_t &t)
{
    check(t.dtype == idx_type<T>::type);
    return ranked_tensor_ref_t<T, r>(ranked<r>(t.shape), (T *)t.data);
}

template <typename T, uint8_t r>
uint32_t len(const ranked_tensor_ref_t<T, r> &tensor)
{
    return std::get<0>(tensor.shape.dims);
}

template <typename T, uint8_t r>
uint32_t wid(const ranked_tensor_ref_t<T, r> &tensor)
{
    return std::get<1>(tensor.shape.dims);
}

template <typename T> using matrix_ref_t = ranked_tensor_ref_t<T, 2>;

template <typename T> using vector_ref_t = ranked_tensor_ref_t<T, 1>;

template <typename T> vector_ref_t<T> flatten(const tensor_ref_t &r)
{
    check(r.dtype == idx_type<T>::type);
    return vector_ref_t<T>(ranked_shape_t<1>(r.shape.dim()), (T *)r.data);
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

template <typename T> std::string summary(const r_tensor_ref_t<T> &r)
{
    constexpr const char *const fmt = "min: %12f    meam: %12f    max: %12f";
    char line[256];
    sprintf(line, fmt, r.min(), r.mean(), r.max());
    return line;
}

template <typename T> void print(const r_tensor_ref_t<T> &r)
{
    printf("%s\n", summary(r).c_str());
}
