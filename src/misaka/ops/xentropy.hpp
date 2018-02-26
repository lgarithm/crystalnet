#pragma once
#include <misaka.h>
#include <misaka/core/debug.hpp>
#include <misaka/model/operator.hpp>

struct xentropy {
    constexpr static uint8_t arity = 2;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        return new shape_t();
    }

    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            assert(inputs.arity() == arity);
            auto a = as_vector_ref<T>(inputs[0]);
            auto b = as_vector_ref<T>(inputs[1]);
            auto c = r_tensor_ref_t<T>(output);
            auto n = equally(a.n, b.n);
            T z = 0;
            for (auto i = 0; i < n; ++i) {
                z += a(i) * log(b(i));
            }
            c.data[0] = -z;
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const T g = r_tensor_ref_t<T>(output_gradient).data[0];
            const auto x = as_vector_ref<T>(inputs[0]);
            const auto y = as_vector_ref<T>(inputs[1]);
            const auto gx = as_vector_ref<T>(input_gradients[0]);
            const auto gy = as_vector_ref<T>(input_gradients[1]);
            const auto n = equally(x.n, y.n); // == gx.n == gy.n
            for (auto i = 0; i < n; ++i) {
                gx.data[i] = g * -log(y.data[i]);
                gy.data[i] = g * (-x.data[i] / y.data[i]);
            }
        }
    };
};

operator_t *op_xentropy = _register_bi_op<xentropy>("cross entropy");
