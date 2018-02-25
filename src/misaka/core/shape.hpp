#pragma once
#include <cstdint>

#include <string>
#include <vector>

struct shape_t {
    shape_t(uint8_t rank) : dims(rank)
    {
        std::fill(dims.begin(), dims.end(), 1);
    }

    shape_t(const shape_t &other) : dims(other.dims) {}

    explicit shape_t(const std::vector<uint32_t> &dims) : dims(dims) {}

    // TODO: more constructors

    uint8_t rank() const { return dims.size(); }

    uint32_t dim() const
    {
        // TODO: use std::reduce when it's available
        uint32_t d = 1;
        for (auto i : dims) {
            d *= i;
        }
        return d;
    }

    uint32_t len() const
    {
        if (dims.size() > 0) {
            return dims[0];
        }
        return 1;
    }

    std::vector<uint32_t> dims; // TODO: make it const
};

struct shape_list_t {
    std::vector<shape_t> shapes;
};

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
