#pragma once
#include <crystalnet.h>
#include <crystalnet/core/debug.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/model/operator.hpp>

struct mul_vm {
    constexpr static uint8_t arity = 2;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        auto p = (*shape_list)[0];
        auto q = (*shape_list)[1];
        assert(p.rank() == 1);
        assert(q.rank() == 2);
        assert(p.dims[0] == q.dims[0]);
        return new shape_t(q.dims[1]);
    }

    using T = float; // TODO: cast based on dtype

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
