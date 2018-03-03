#pragma once
#include <memory>
#include <crystalnet.h>
#include <crystalnet/core/shape.hpp>
#include <vector>

struct placeholder_t {
    const shape_t shape;
    explicit placeholder_t(const shape_t &shape) : shape(shape) {}
};
