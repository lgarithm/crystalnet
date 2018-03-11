#pragma once
#include <crystalnet/core/initializer.hpp>

struct constant_initializer_t : initializer_t {
    const double c;
    explicit constant_initializer_t(double c) : c(c) {}
    void operator()(const tensor_ref_t &t) const override
    {
        using T = float;
        r_tensor_ref_t<T> r(t);
        r.fill(c);
    }
};