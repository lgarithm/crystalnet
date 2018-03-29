#pragma once
#include <cmath>

#include <crystalnet.h>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/ops/batch.hpp>
#include <crystalnet/utility/cast.hpp>
#include <crystalnet/utility/range.hpp>

template <typename T>
void softmax_eval_safe(const vector_ref_t<T> &input,
                       const vector_ref_t<T> &output)
{
    const auto n = len(input);
    for (auto i : range(n)) {
        T tot = 0;
        for (auto j : range(n)) {
            tot += exp(input.data[j] - input.data[i]);
        }
        output.data[i] = std::max((T)1e-6, (T)1.0 / tot);
    }
}

template <typename T>
void softmax_grad(const vector_ref_t<T> &output, const matrix_ref_t<T> &grad)
{
    const auto n = len(output);
    for (auto i : range(n)) {
        grad.data[i * n + i] = output.data[i] * (1 - output.data[i]);
    }
    for (auto i : range(n)) {
        for (auto j : range(i)) {
            grad.data[i * n + j] = grad.data[j * n + i] =
                -output.data[i] * output.data[j];
        }
    }
}

struct softmax_1d {
    constexpr static uint8_t arity = 1;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p] = cast<arity>(shape_list.shapes);
        return p;
    }

    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            check(inputs.arity() == arity);
            softmax_eval_safe(ranked<1, T>(inputs[0]), ranked<1, T>(output));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            auto n = output.shape.dim();
            tensor_t tmp(shape_t(n, n), idx_type<T>::type);
            softmax_grad(ranked<1, T>(output), ranked<2, T>(ref(tmp)));
            linag<T>::vm(ranked<1, T>(output_gradient), ranked<2, T>(ref(tmp)),
                         ranked<1, T>(input_gradients[0]));
        }
    };
};

struct softmax {
    constexpr static uint8_t arity = 1;
    using softmax_2d = batch<softmax_1d, 0>;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p] = cast<arity>(shape_list.shapes);
        return p;
    }

    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto[p] = cast<1>(inputs.shapes().shapes);
            if (p.rank() == 1) {
                (*(softmax_1d::forward *)this)();
            } else {
                check(p.rank() == 2);
                (*(softmax_2d::forward *)this)();
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto[p] = cast<1>(inputs.shapes().shapes);
            if (p.rank() == 1) {
                (*(softmax_1d::backward *)this)();
            } else {
                check(p.rank() == 2);
                (*(softmax_2d::backward *)this)();
            }
        }
    };
};
