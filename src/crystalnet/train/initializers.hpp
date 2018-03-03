#pragma once

#include <cmath>
#include <random>

#include <crystalnet/core/tensor.hpp>

template <typename T> struct constant_initializer {
    const T c;

    explicit constant_initializer(T c) : c(c) {}

    void operator()(const r_tensor_ref_t<T> &r) const { r.fill(c); }
};

template <typename T> struct truncated_normal_initializer {
    const T stddev;
    const T _bound;

    explicit truncated_normal_initializer(T stddev)
        : stddev(stddev), _bound(2 * std::fabs(stddev))
    {
    }

    void operator()(const r_tensor_ref_t<T> &r) const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> d(0, stddev);
        const auto n = r.shape.dim();
        for (auto i = 0; i < n; ++i) {
            T x = d(gen);
            while (std::fabs(x) > _bound) {
                x = d(gen);
            }
            r.data[i] = x;
        }
    }
};
