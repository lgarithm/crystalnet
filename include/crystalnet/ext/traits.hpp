#pragma once
#include <crystalnet-ext.h>
#include <crystalnet/core/shape.hpp>

struct filter_trait_t {
    const shape_t shape;
    explicit filter_trait_t(const shape_t &shape) : shape(shape) {}
};

struct padding_trait_t {
    const shape_t shape;
    explicit padding_trait_t(const shape_t &shape) : shape(shape) {}
};

struct stride_trait_t {
    const shape_t shape;
    explicit stride_trait_t(const shape_t &shape) : shape(shape) {}
};
