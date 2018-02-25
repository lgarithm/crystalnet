#pragma once
#include <memory>
#include <misaka.h>
#include <misaka/core/shape.hpp>
#include <vector>

struct placeholder_t {
    const shape_t shape;
    explicit placeholder_t(const shape_t &shape) : shape(shape) {}
};
