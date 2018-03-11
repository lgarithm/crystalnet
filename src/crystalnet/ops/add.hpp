#pragma once
#include <crystalnet.h>
#include <crystalnet/core/debug.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/model/operator.hpp>
#include <crystalnet/ops/batch.hpp>

// [n], [n] -> [n]
struct add_vv {
    constexpr static uint8_t arity = 2;

    static shape_t *infer(const shape_list_t *shapes)
    {
        assert(shapes->size() == arity);
        return new shape_t((*shapes)[0]);
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            linag<T>::vv(as_vector_ref<T>(inputs[0]),
                         as_vector_ref<T>(inputs[1]), as_vector_ref<T>(output));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            auto g = r_tensor_ref_t<T>(output_gradient);
            r_tensor_ref_t<T>(input_gradients[0]).copy(g);
            r_tensor_ref_t<T>(input_gradients[1]).copy(g);
        }
    };
};

struct add {
    constexpr static uint8_t arity = 2;
    using add_mv = batch<add_vv, 0>;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        const auto[p, q] = cast<2>(shape_list->shapes);
        if (p.rank() > q.rank()) {
            return add_mv::infer(shape_list);
        }
        assert(p.rank() == q.rank());
        return add_vv::infer(shape_list);
    }

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<2>(inputs.shapes().shapes);
            if (p.rank() > q.rank()) {
                (*(add_mv::forward *)this)();
            } else {
                assert(p.rank() == q.rank());
                (*(add_vv::forward *)this)();
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<2>(inputs.shapes().shapes);
            if (p.rank() > q.rank()) {
                (*(add_mv::backward *)this)();
            } else {
                assert(p.rank() == q.rank());
                (*(add_vv::backward *)this)();
            }
        }
    };
};

operator_t *op_add = _register_bi_op<add>("add");
