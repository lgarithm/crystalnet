#pragma once

#include <cmath>

#include <crystalnet-internal.h>
#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/operator.hpp>  // TODO: don't include private headers
#include <crystalnet/layers/layer.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/utility/range.hpp>

namespace darknet
{
template <typename T> struct op_batch_norm {
    constexpr static uint8_t arity = 3;

    shape_t _infer(const shape_list_t &shape_list) const
    {
        const auto [p, q, r] = cast<arity>(shape_list.shapes, auto_hint);
        const auto [b, c, h, w] = cast<4>(p.dims, auto_hint);
        check(q == shape_t(c));
        check(r == shape_t(c));
        return p;
    }

    // bn :: [n, c, h, w], [c], [c] -> [n, c, h, w]
    shape_t infer(const shape_list_t &shape_list) const
    {
        const auto [p, q, r] = cast<arity>(shape_list.shapes, auto_hint);
        if (p.rank() == 3) {
            return _infer(shape_list_t({p.batch(1), q, r}));
        } else {
            return _infer(shape_list);
        }
    }

    void forward(const forward_ctx_t &ctx) const
    {
        const T eps = .000001f;
        const auto [_x, rolling_means, rolling_variance] =
            cast<arity>(ctx.inputs._args, auto_hint);

        const auto x = ranked<4, T>(_x);
        const auto rm = ranked<1, T>(rolling_means);
        const auto rv = ranked<1, T>(rolling_variance);
        const auto y = ranked<4, T>(ctx.output);

        const auto [n, c, h, w] = cast<4>(_x.shape.dims, auto_hint);

        for (const auto b : range(n)) {
            for (const auto l : range(c)) {
                for (const auto i : range(h)) {
                    for (const auto j : range(w)) {
                        y.at(b, l, i, j) = (x.at(b, l, i, j) - rm.at(l)) /
                                           (std::sqrt(rv.at(l)) + eps);
                    }
                }
            }
        }
    }

    void backward(const backward_ctx_t &ctx) const
    {
        throw std::logic_error("NOT IMPLEMENTED");
    }
};
}  // namespace darknet
