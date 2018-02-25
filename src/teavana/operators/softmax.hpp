#pragma once

#include <cmath>   // for exp
#include <cstddef> // for size_t
#include <cstdint> // for uint8_t
#include <utility> // for integer_sequence

#include "teavana/core/tensor.hpp" // for tensor_ref (ptr only), ref, tensor
#include "teavana/operator.hpp"    // for in, gin, select_ctx
#include "teavana/range.hpp"       // for range

namespace tea
{
template <uint8_t r> struct shape_t;
} // namespace tea

namespace tea
{
template <typename R>
void softmax_eval_with_tmp(size_t n, const tensor_ref<R, 1> &input,
                           const tensor_ref<R, 1> &output,
                           const tensor_ref<R, 1> &tmp)
{
    R tot = 0;
    for (auto i : range(n)) {
        tmp.data[i] = exp(input.data[i]);
        tot += tmp.data[i];
    }
    for (auto i : range(n)) {
        output.data[i] = tmp.data[i] / tot;
    }
}

template <typename R>
void softmax_eval_safe(size_t n, const tensor_ref<R, 1> &input,
                       const tensor_ref<R, 1> &output)
{
    for (auto i : range(n)) {
        R tot = 0;
        for (auto j : range(n)) {
            tot += exp(input.data[j] - input.data[i]);
        }
        output.data[i] = ::std::max((R)1e-6, (R)1.0 / tot);
    }
}

template <typename R>
void softmax_grad(size_t n, const tensor_ref<R, 1> & /* input */,
                  const tensor_ref<R, 1> &output, const tensor_ref<R, 2> &grad)
{
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

struct op_softmax {
    static constexpr const char *name = "softmax";
    using signature = ::std::index_sequence<1, 1>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<1> &s) { return s; }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;

        using t_p = typename ctx_types::template temp_ctx_type<R>;

        struct temp_ctx_t : t_p {
            const tensor<R, 1> tmp_;
            const tensor<R, 2> grad1;
            temp_ctx_t(const eval_ctx_t &ctx)
                : t_p(ctx), tmp_(ctx.output.shape),
                  grad1(ctx.output.shape * in<0>(ctx).shape)
            {
            }
        };

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            softmax_eval_with_tmp(dim(ctx.output.shape), in<0>(ctx), ctx.output,
                                  ref(t_ctx.tmp_));
            // softmax_eval_safe(dim(ctx.output.shape), in<0>(ctx), ctx.output);
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            softmax_grad(dim(ctx.output.shape), in<0>(ctx), ctx.output,
                         ref(t_ctx.grad1));
            vec_x_mat(as_vector(ctx.g_output), ref(t_ctx.grad1),
                      as_vector(gin<0>(ctx)));
        }
    };
};
}
