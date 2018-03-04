#pragma once
#include <memory>
#include <vector>

#include <crystalnet.h>
#include <crystalnet/core/shape.hpp>

struct placeholder_t {
    const shape_t shape;
    explicit placeholder_t(const shape_t &shape) : shape(shape) {}
};
