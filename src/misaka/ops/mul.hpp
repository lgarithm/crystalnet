#pragma once
#include <misaka.h>
#include <misaka/core/debug.hpp>
#include <misaka/linag/linag.hpp>
#include <misaka/model/operator.hpp>

struct mul_vm {
    constexpr static uint8_t arity = 2;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        auto s1 = shape_list->shapes[0];
        auto s2 = shape_list->shapes[1];
        assert(s1.rank() == 1);
        assert(s2.rank() == 2);
        assert(s1.dims[0] == s2.dims[0]);
        return make_shape(1, s2.dims[1]);
    }

    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            linag<T>::vm(as_vector_ref<T>(inputs[0]),
                         as_matrix_ref<T>(inputs[1]), as_vector_ref<T>(output));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            linag<T>::mv(as_matrix_ref<T>(inputs[1]),
                         as_vector_ref<T>(output_gradient),
                         as_vector_ref<T>(input_gradients[0]));
            linag<T>::mm(as_col_matrix_ref(as_vector_ref<T>(inputs[0])),
                         as_row_matrix_ref(as_vector_ref<T>(output_gradient)),
                         as_matrix_ref<T>(input_gradients[1]));
        }
    };
};

operator_t *op_mul = _register_bi_op<mul_vm>("mul_vm");
