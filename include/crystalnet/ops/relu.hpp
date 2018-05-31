#pragma once
#include <crystalnet.h>
#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/utility/range.hpp>

struct relu {
    constexpr static uint8_t arity = 1;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p] = cast<arity>(shape_list.shapes, auto_hint);
        return p;
    }

    using T = float; // TODO: cast based on dtype

    template <typename T> inline static T _relu(T x) { return x > 0 ? x : 0; }

    struct forward : forward_ctx_t {
        void operator()() const
        {
            auto x = flatten<T>(inputs[0]);
            auto y = flatten<T>(output);
            check(len(x) == len(y));
            for (auto i : range(len(x))) {
                y.data[i] = _relu(x.data[i]);
            }
        }
    };

    template <typename T> inline static T _relu_grad(T x)
    {
        return x > 0 ? 1.0 : 0.0;
    }

    struct backward : backward_ctx_t {
        void operator()() const
        {
            auto x = flatten<T>(inputs[0]);
            auto gx = flatten<T>(input_gradients[0]);
            auto gy = flatten<T>(output_gradient);
            check(len(x) == len(gx));
            check(len(gx) == len(gy));
            for (auto i : range(len(x))) {
                gx.data[i] = gy.data[i] * _relu_grad(x.data[i]);
            }
        }
    };
};
