#pragma once
#include <memory>
#include <vector>

#include <crystalnet.h>
#include <crystalnet/core/shape.hpp>

struct parameter_t {
    const shape_t shape;
    explicit parameter_t(const shape_t &shape) : shape(shape) {}
};
