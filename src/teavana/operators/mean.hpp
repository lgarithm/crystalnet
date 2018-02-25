#pragma once

#include <utility> // for integer_sequence

#include "teavana/core/shape.hpp"  // for shape, dim, shape_t (ptr only)
#include "teavana/core/tensor.hpp" // for scalar, fill
#include "teavana/operator.hpp"    // for gin, in, select_ctx
#include "teavana/range.hpp"       // for range

namespace tea
{
struct op_mean {
    static constexpr const char *name = "mean";
    using signature = ::std::index_sequence<0, 1>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<1> &) { return shape(); }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto n = dim(x.shape);
            R tmp = 0;
            for (auto i : range(n)) {
                tmp += x.data[i];
            }
            scalar(ctx.output) = tmp / n;
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto gx = gin<0>(ctx);
            fill(gx, scalar(ctx.g_output) / dim(gx.shape));
        }
    };
};
}
