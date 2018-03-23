#pragma once
#include <crystalnet.h>
#include <crystalnet/core/debug.hpp>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/utility/cast.hpp>

template <typename T> matrix_ref_t<T> col(const vector_ref_t<T> &r)
{
    return matrix_ref_t<T>(ranked_shape_t<2>(r.shape.dim(), 1), r.data);
}

template <typename T> matrix_ref_t<T> row(const vector_ref_t<T> &r)
{
    return matrix_ref_t<T>(ranked_shape_t<2>(1, r.shape.dim()), r.data);
}

struct mul_vm {
    constexpr static uint8_t arity = 2;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list.shapes);
        const auto[m] = cast<1>(p.dims);
        const auto[_m, n] = cast<2>(q.dims);
        check(m == _m);
        return shape_t(n);
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            linag<T>::vm(ranked<1, T>(inputs[0]), ranked<2, T>(inputs[1]),
                         ranked<1, T>(output));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            linag<T>::mv(ranked<2, T>(inputs[1]), ranked<1, T>(output_gradient),
                         ranked<1, T>(input_gradients[0]));
            linag<T>::mm(col(ranked<1, T>(inputs[0])),
                         row(ranked<1, T>(output_gradient)),
                         ranked<2, T>(input_gradients[1]));
        }
    };
};

struct mul_mm {
    constexpr static uint8_t arity = 2;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list.shapes);
        const auto[k, m] = cast<2>(p.dims);
        const auto[_m, n] = cast<2>(q.dims);
        check(m == _m);
        return shape_t(k, n);
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            linag<T>::mm(ranked<2, T>(inputs[0]), ranked<2, T>(inputs[1]),
                         ranked<2, T>(output));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            linag<T>::mmt(ranked<2, T>(output_gradient),
                          ranked<2, T>(inputs[1]),
                          ranked<2, T>(input_gradients[0]));
            linag<T>::mtm(ranked<2, T>(inputs[0]),
                          ranked<2, T>(output_gradient),
                          ranked<2, T>(input_gradients[1]));
        }
    };
};

struct mul {
    constexpr static uint8_t arity = 2;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list.shapes);
        if (p.rank() == 1 && q.rank() == 2) {
            return mul_vm::infer(shape_list);
        }
        check(p.rank() == 2 && q.rank() == 2);
        return mul_mm::infer(shape_list);
    }

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<arity>(inputs.shapes().shapes);
            if (p.rank() == 1 && q.rank() == 2) {
                (*(mul_vm::forward *)this)();
            } else {
                check(p.rank() == 2 && q.rank() == 2);
                (*(mul_mm::forward *)this)();
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<arity>(inputs.shapes().shapes);
            if (p.rank() == 1 && q.rank() == 2) {
                (*(mul_vm::backward *)this)();
            } else {
                check(p.rank() == 2 && q.rank() == 2);
                (*(mul_mm::backward *)this)();
            }
        }
    };
};
