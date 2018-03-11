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
        const auto[m] = cast<1>((*shape_list)[0].dims);
        const auto[_m, n] = cast<2>((*shape_list)[1].dims);
        assert(m == _m);
        return new shape_t(n);
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

struct mul_mm {
    constexpr static uint8_t arity = 2;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        const auto[k, m] = cast<2>((*shape_list)[0].dims);
        const auto[_m, n] = cast<2>((*shape_list)[1].dims);
        assert(m == _m);
        return new shape_t(k, n);
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            linag<T>::mm(as_matrix_ref<T>(inputs[0]),
                         as_matrix_ref<T>(inputs[1]), as_matrix_ref<T>(output));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            linag<T>::mmt(as_matrix_ref<T>(output_gradient),
                          as_matrix_ref<T>(inputs[1]),
                          as_matrix_ref<T>(input_gradients[0]));
            linag<T>::mtm(as_matrix_ref<T>(inputs[0]),
                          as_matrix_ref<T>(output_gradient),
                          as_matrix_ref<T>(input_gradients[1]));
        }
    };
};

struct mul {
    constexpr static uint8_t arity = 2;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        const auto[p, q] = cast<2>(shape_list->shapes);
        if (p.rank() == 1 && q.rank() == 2) {
            return mul_vm::infer(shape_list);
        }
        assert(p.rank() == 2 && q.rank() == 2);
        return mul_mm::infer(shape_list);
    }

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<2>(inputs.shapes().shapes);
            if (p.rank() == 1 && q.rank() == 2) {
                (*(mul_vm::forward *)this)();
            } else {
                assert(p.rank() == 2 && q.rank() == 2);
                (*(mul_mm::forward *)this)();
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<2>(inputs.shapes().shapes);
            if (p.rank() == 1 && q.rank() == 2) {
                (*(mul_vm::backward *)this)();
            } else {
                assert(p.rank() == 2 && q.rank() == 2);
                (*(mul_mm::backward *)this)();
            }
        }
    };
};

operator_t *op_mul = _register_bi_op<mul>("mul");
