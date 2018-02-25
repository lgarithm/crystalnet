#pragma once

#include <cassert> // for assert
#include <cstddef> // for size_t
#include <cstdint> // for uint8_t
#include <utility> // for integer_sequence

#include "teavana/core/tensor.hpp" // for scalar, assign
#include "teavana/operator.hpp"    // for gin, in, select_ctx
#include "teavana/range.hpp"       // for range

namespace tea
{
template <uint8_t r> struct shape_t;
} // namespace tea

namespace tea
{
template <template <typename, uint8_t> class T, typename R, uint8_t r>
void incr(const T<R, r> &s, const T<R, r> &t, R x)
{
    assert(s.shape == t.shape);
    const size_t n = dim(s.shape);
    for (size_t i = 0; i < n; ++i) {
        s.data[i] = t.data[i] + x;
    }
}

struct op_incr2d {
    static constexpr const char *name = "incr";
    using signature = ::std::index_sequence<2, 2, 0>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<2> &s1, const shape_t<0> &)
    {
        return s1;
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);
            incr(ctx.output, x, scalar(y));
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            assign(gin<0>(ctx), ctx.g_output);
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            R tmp = 0;
            for (auto i : range(dim(ctx.g_output.shape))) {
                tmp += ctx.g_output.data[i];
            }
            scalar(gin<1>(ctx)) = tmp;
        }
    };
};
}
