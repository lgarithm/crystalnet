#pragma once

#include <cassert> // for assert
#include <utility> // for integer_sequence

#include "teavana/core/shape.hpp" // for len, dim, operator==, shape, sub, shape_t
#include "teavana/core/tensor.hpp" // for ref_as, tensor
#include "teavana/operator.hpp"    // for in, gin, select_ctx
#include "teavana/range.hpp"       // for range

namespace tea
{
template <typename R> R sqr(R x) { return x * x; }

struct op_rbf {
    static constexpr const char *name = "rbf";
    using signature = ::std::index_sequence<1, 2, 1>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<2> &s1,
                                      const shape_t<1> &s2)
    {
        assert(sub(s1) == s2 && __FILE__);
        return shape(len(s1));
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;

        using t_p = typename ctx_types::template temp_ctx_type<R>;

        struct temp_ctx_t : t_p {
            const tensor<R, 3> grad1;
            const tensor<R, 2> grad2;
            temp_ctx_t(const eval_ctx_t &ctx)
                : t_p(ctx), grad1(ctx.output.shape * in<0>(ctx).shape),
                  grad2(ctx.output.shape * in<1>(ctx).shape)
            {
            }
        };

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);

            for (auto i : range(len(ctx.output))) {
                R tmp = 0;
                for (auto j : range(len(y))) {
                    tmp += sqr(x.at(i, j) - y.data[j]);
                }
                ctx.output.at(i) = tmp;
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);

            for (auto i : range(len(ctx.output))) {
                for (auto j : range(len(y))) {
                    t_ctx.grad1.at(i, i, j) = 2 * (x.at(i, j) - y.data[j]);
                }
            }

            vec_x_mat(
                as_vector(ctx.g_output),
                ref_as(t_ctx.grad1, shape(dim(ctx.output.shape), dim(x.shape))),
                as_vector(gin<0>(ctx)));
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);

            for (auto i : range(len(ctx.output))) {
                for (auto j : range(len(y))) {
                    t_ctx.grad2.at(i, j) = 2 * (y.data[j] - x.at(i, j));
                }
            }

            vec_x_mat(
                as_vector(ctx.g_output),
                ref_as(t_ctx.grad2, shape(dim(ctx.output.shape), dim(y.shape))),
                as_vector(gin<1>(ctx)));
        }
    };
};
}
