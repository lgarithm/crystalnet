#pragma once
#include <memory>
#include <misaka.h>
#include <misaka/core/shape.hpp>
#include <vector>

struct parameter_t {
    const shape_t shape;
    explicit parameter_t(const shape_t &shape) : shape(shape) {}
};
