#pragma once

#include <crystalnet-internal.h>
#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/layer.hpp>
#include <crystalnet/core/operator.hpp>  // TODO: don't include private headers
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/utility/range.hpp>

namespace darknet
{
template <typename Op> struct op_bias {
    constexpr static uint8_t arity = 2;

    shape_t _infer(const shape_list_t &shape_list) const
    {
        const auto [p, q] = cast<arity>(shape_list.shapes, auto_hint);
        const auto [b, c, h, w] = cast<4>(p.dims, auto_hint);
        check(q == shape_t(c));
        return p;
    }

    // bias :: [n, c, h, w], [c] -> [n, c, h, w]
    shape_t infer(const shape_list_t &shape_list) const
    {
        const auto [p, q] = cast<arity>(shape_list.shapes, auto_hint);
        if (p.rank() == 3) {
            return _infer(shape_list_t({p.batch(1), q}));
        } else {
            return _infer(shape_list);
        }
    }

    void forward(const forward_ctx_t &ctx) const
    {
        const Op op;

        using T = float;
        const auto [_x, _bias] = cast<arity>(ctx.inputs._args, auto_hint);
        const auto x = ranked<4, T>(_x);
        const auto bias = ranked<1, T>(_bias);
        const auto y = ranked<4, T>(ctx.output);
        const auto [n, c, h, w] = x.shape.dims;
        for (auto b : range(n)) {
            for (auto l : range(c)) {
                for (auto i : range(h)) {
                    for (auto j : range(w)) {
                        y.at(b, l, i, j) = op(x.at(b, l, i, j), bias.at(l));
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

using add_bias = op_bias<std::plus<float>>;
using scale_bias = op_bias<std::multiplies<float>>;

}  // namespace darknet
