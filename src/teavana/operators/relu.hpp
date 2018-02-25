#pragma once

#include <cstdint> // for uint8_t
#include <utility> // for integer_sequence

#include "teavana/operator.hpp" // for in, gin, select_ctx
#include "teavana/range.hpp"    // for range

namespace tea
{
template <uint8_t r> struct shape_t;
} // namespace tea

namespace tea
{
template <typename R> R relu_eval_scalar(R x) { return x > 0 ? x : 0.0; }

template <typename R> R relu_grad_scalar(R x) { return x > 0 ? 1.0 : 0.0; }

struct op_relu {
    static constexpr const char *name = "relu";
    using signature = ::std::index_sequence<1, 1>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<1> &s) { return s; }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            for (auto i : range(dim(ctx.output.shape))) {
                ctx.output.data[i] = relu_eval_scalar(x.data[i]);
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto gx = gin<0>(ctx);
            const auto gy = ctx.g_output;
            for (auto i : range(dim(ctx.output.shape))) {
                gx.data[i] = gy.data[i] * relu_grad_scalar(x.data[i]);
            }
        }
    };
};
}
