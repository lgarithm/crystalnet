#pragma once
#include <misaka.h>
#include <misaka/core/debug.hpp>
#include <misaka/core/shape.hpp>
#include <misaka/linag/linag.hpp>
#include <misaka/model/operator.hpp>

struct add {
    constexpr static uint8_t arity = 2;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        auto shape = new shape_t(shape_list->shapes[0]);
        return shape;
    }

    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            assert(inputs.arity() == arity);
            linag<T>::vv(as_vector_ref<T>(inputs[0]),
                         as_vector_ref<T>(inputs[1]), as_vector_ref<T>(output));
        }
    };

    template <typename T>
    static void assign(const vector_ref_t<T> &a, const vector_ref_t<T> &b)
    {
        auto n = equally(a.n, b.n);
        memcpy(a.data, b.data, n * sizeof(T));
    }

    struct backward : backward_ctx_t {
        void operator()() const
        {
            auto g = as_vector_ref<T>(output_gradient);
            assign(as_vector_ref<T>(input_gradients[0]), g);
            assign(as_vector_ref<T>(input_gradients[1]), g);
        }
    };
};

operator_t *op_add = _register_bi_op<add>("add");
