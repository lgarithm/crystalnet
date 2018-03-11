#pragma once
#include <cassert>
#include <cstdint>

#include <array>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <crystalnet/core/gc.hpp>

struct shape_t {
    const std::vector<uint32_t> dims;

    template <typename... T>
    explicit shape_t(T... dims) : dims({static_cast<uint32_t>(dims)...})
    {
    }

    explicit shape_t(const std::vector<uint32_t> &dims) : dims(dims) {}

    uint8_t rank() const { return dims.size(); }

    uint32_t dim() const
    {
        return std::accumulate(dims.begin(), dims.end(), 1,
                               std::multiplies<uint32_t>());
    }

    uint32_t len() const
    {
        if (dims.size() > 0) {
            return dims[0];
        }
        return 1;
    }

    shape_t sub() const
    {
        if (dims.size() > 0) {
            return shape_t(std::vector<uint32_t>(dims.begin() + 1, dims.end()));
        }
        return *this;
    }

    shape_t batch(uint32_t n) const
    {
        std::vector<uint32_t> dims({n});
        dims.insert(dims.end(), this->dims.begin(), this->dims.end());
        return shape_t(dims);
    }
};

struct shape_ctx_t {
    GC<shape_t> gc;
    const shape_t *make_shape(const std::vector<uint32_t> &dims)
    {
        return gc(new shape_t(dims));
    }
};

struct shape_list_t {
    const std::vector<shape_t> shapes;
    explicit shape_list_t(const std::vector<shape_t> &shapes) : shapes(shapes)
    {
    }
    shape_t operator[](int i) const { return shapes[i]; }
    uint8_t size() const { return shapes.size(); }
};

template <typename T, size_t... i>
auto _index(const std::vector<T> &v, std::index_sequence<i...>)
{
    return std::array<T, sizeof...(i)>({v[i]...});
}

template <uint8_t rank, typename T> auto cast(const std::vector<T> &v)
{
    assert(v.size() == rank);
    return _index(v, std::make_index_sequence<rank>());
}

namespace std
{
inline string to_string(const shape_t &shape)
{
    string buf;
    for (auto d : shape.dims) {
        if (!buf.empty()) {
            buf += ",";
        }
        buf += to_string(d);
    }
    return "shape(" + buf + ")";
}
}
