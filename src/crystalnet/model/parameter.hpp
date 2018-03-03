#pragma once
#include <memory>
#include <crystalnet.h>
#include <crystalnet/core/shape.hpp>
#include <vector>

struct parameter_t {
    const shape_t shape;
    explicit parameter_t(const shape_t &shape) : shape(shape) {}
};
