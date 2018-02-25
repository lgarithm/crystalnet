#pragma once

#include <array>   // for array
#include <cassert> // for assert
#include <cstddef> // for size_t
#include <utility> // for integer_sequence

#include "teavana/core/shape.hpp"  // for shape_t, shape
#include "teavana/core/tensor.hpp" // for annihilate
#include "teavana/operator.hpp"    // for gin, in, select_ctx
#include "teavana/range.hpp"       // for range

namespace tea
{
struct op_pool2d_max {
    static constexpr const char *name = "pool2d_max";
    using signature = ::std::index_sequence<2, 2>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<2> &s1)
    {
        assert(s1.dims[0] % 2 == 0);
        assert(s1.dims[1] % 2 == 0);
        return shape(s1.dims[0] / 2, s1.dims[1] / 2);
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto y = ctx.output;
            annihilate(y);
            for (auto i1 : range(dim<0>(x))) {
                for (auto i2 : range(dim<1>(x))) {
                    R &yy = y.at(i1 / 2, i2 / 2);
                    yy = ::std::max(yy, x.at(i1, i2));
                }
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto gx = gin<0>(ctx);
            const auto gy = ctx.g_output;
            for (auto j1 : range(dim<0>(gy))) {
                for (auto j2 : range(dim<1>(gy))) {
                    for (auto i1 : range(j1 * 2, j1 * 2 + 2)) {
                        for (auto i2 : range(j2 * 2, j2 * 2 + 2)) {
                            gx.at(i1, i2) = gy.at(j1, j2);
                        }
                    }
                }
            }
        }
    };
};

struct op_pool2d_c_max {
    static constexpr const char *name = "pool2d_c_max";
    using signature = ::std::index_sequence<3, 3>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<3> &s1)
    {
        assert(s1.dims[0] % 2 == 0);
        assert(s1.dims[1] % 2 == 0);
        return shape(s1.dims[0] / 2, s1.dims[1] / 2, s1.dims[2]);
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto y = ctx.output;
            annihilate(y);
            for (auto i1 : range(dim<0>(x))) {
                for (auto i2 : range(dim<1>(x))) {
                    for (auto i3 : range(dim<1>(x))) {
                        R &yy = y.at(i1 / 2, i2 / 2, i3);
                        yy = ::std::max(yy, x.at(i1, i2, i3));
                    }
                }
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto gx = gin<0>(ctx);
            const auto gy = ctx.g_output;
            for (auto j1 : range(dim<0>(gy))) {
                for (auto j2 : range(dim<1>(gy))) {
                    for (auto i1 : range(j1 * 2, j1 * 2 + 2)) {
                        for (auto i2 : range(j2 * 2, j2 * 2 + 2)) {
                            for (auto i3 : range(dim<2>(gx))) {
                                gx.at(i1, i2, i3) = gy.at(j1, j2, i3);
                            }
                        }
                    }
                }
            }
        }
    };
};
}
