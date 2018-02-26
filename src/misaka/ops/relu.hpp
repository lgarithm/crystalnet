#pragma once
#include <misaka.h>
#include <misaka/model/operator.hpp>

struct relu {
    constexpr static uint8_t arity = 1;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        return new shape_t(shape_list->shapes[0]);
    }

    using T = float;

    template <typename T> inline static T _relu(T x) { return x > 0 ? x : 0; }

    struct forward : forward_ctx_t {
        void operator()() const
        {
            // TODO: case based on inputs.dtype
            auto x = as_vector_ref<T>(inputs[0]);
            auto y = as_vector_ref<T>(output);
            auto n = equally(x.n, y.n);
            for (auto i = 0; i < n; ++i) {
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
            auto x = as_vector_ref<T>(inputs[0]);
            auto gx = as_vector_ref<T>(input_gradients[0]);
            auto gy = as_vector_ref<T>(output_gradient);
            auto n = x.n; // == gx.n == gy.n
            for (auto i = 0; i < n; ++i) {
                gx.data[i] = gy.data[i] * _relu_grad(x.data[i]);
            }
        }
    };
};

operator_t *op_relu = _register_bi_op<relu>("relu");
