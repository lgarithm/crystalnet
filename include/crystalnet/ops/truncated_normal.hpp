#pragma once
#include <algorithm>
#include <cmath>
#include <random>

#include <crystalnet/core/operator.hpp>

struct truncated_normal_initializer_t : initializer_t {
    const double stddev;
    const double _bound;

    explicit truncated_normal_initializer_t(double stddev)
        : stddev(stddev), _bound(2 * std::fabs(stddev))
    {
    }

    void operator()(const tensor_ref_t &t) const override
    {
        using T = float;
        r_tensor_ref_t<T> r(t);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> d(0, stddev);
        std::generate(r.data, r.data + r.shape.dim(), [&]() {
            T x = d(gen);
            while (std::fabs(x) > _bound) {
                x = d(gen);
            }
            return x;
        });
    }
};