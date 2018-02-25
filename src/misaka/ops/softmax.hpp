#pragma once
#include <misaka.h>
#include <misaka/core/debug.hpp>
#include <misaka/model/operator.hpp>
#include <teavana/operators/softmax.hpp>

struct softmax {
    constexpr static uint8_t arity = 1;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        auto shape = new shape_t(shape_list->shapes[0]);
        return shape;
    }

    template <typename T>
    static tea::tensor_ref<T, 1> cast(const vector_ref_t<T> &v)
    {
        return tea::tensor_ref<T, 1>(tea::shape(v.n), v.data);
    }

    template <typename T>
    static tea::tensor_ref<T, 2> cast2(const matrix_ref_t<T> &v)
    {
        return tea::tensor_ref<T, 2>(tea::shape(v.m, v.n), v.data);
    }

    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            DEBUG(__func__);
            assert(inputs.arity() == arity);
            auto a = as_vector_ref<T>(inputs[0]);
            auto b = as_vector_ref<T>(output);
            auto n = equally(a.n, b.n);
            tea::softmax_eval_safe(n, cast(a), cast(b));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            auto n = output.shape.dim();
            tensor_t tmp(shape_t(std::vector<uint32_t>({n, n})),
                         idx_type<T>::type);
            tea::softmax_grad(n, //
                              cast(as_vector_ref<T>(inputs[0])),
                              cast(as_vector_ref<T>(output)),
                              cast2(as_matrix_ref<T>(ref(tmp))));
            linag<T>::vm(as_vector_ref<T>(output_gradient),
                         as_matrix_ref<T>(ref(tmp)),
                         as_vector_ref<T>(input_gradients[0]));
        }
    };
};

operator_t *op_softmax = _register_bi_op<softmax>("softmax");
