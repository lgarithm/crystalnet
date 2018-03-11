#pragma once
#include <crystalnet/core/tensor.hpp>

struct initializer_t {
    virtual void operator()(const tensor_ref_t &) const = 0;
    virtual ~initializer_t() {}
};
