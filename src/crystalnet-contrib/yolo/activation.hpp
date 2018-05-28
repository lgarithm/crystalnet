#pragma once
#include <algorithm>
#include <cmath>

#include <crystalnet-internal.h>
#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/operator.hpp>  // TODO: don't include private headers
#include <crystalnet/layers/layer.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/utility/range.hpp>

template <typename T, typename F> void pointwise(T *begin, T *end, const F &f)
{
    std::transform(begin, end, begin, f);
}

template <typename T> struct logistic {
    T operator()(T x) const { return (T)1. / ((T)1. + std::exp(-x)); }
};

template <typename T> struct leaky_relu {
    T operator()(T x) const { return x > 0 ? x : .1 * x; }
};

template <typename T> struct linear {
    T operator()(T x) const { return x; }
};

namespace darknet
{
template <typename Activate> struct pointwise_op {
    constexpr static uint8_t arity = 1;

    shape_t infer(const shape_list_t &shape_list) const
    {
        const auto [p] = cast<arity>(shape_list.shapes, auto_hint);
        return p;
    }

    void forward(const forward_ctx_t &ctx) const
    {
        using T = float;
        const auto [x] = cast<arity>(ctx.inputs._args, auto_hint);

        const auto n = x.shape.dim();
        const auto x_flat = x.reshape(shape_t(n));
        const auto y_flat = ctx.output.reshape(shape_t(n));

        const auto rx = r_tensor_ref_t<T>(x_flat);
        const auto ry = r_tensor_ref_t<T>(y_flat);

        std::transform(rx.data, rx.data + n, ry.data, Activate());
    }

    void backward(const backward_ctx_t &ctx) const
    {
        // Not implemented
    }
};

}  // namespace darknet
