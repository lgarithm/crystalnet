#include <crystalnet-ext.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/core/layer.hpp>
#include <crystalnet/ops/const.hpp>
#include <crystalnet/ops/truncated_normal.hpp>

struct conv_nhwc : s_layer_t {
    const uint32_t r;
    const uint32_t s;
    const uint32_t d;

    conv_nhwc(uint32_t r, uint32_t s, uint32_t d = 1) : r(r), s(s), d(d) {}

    static uint32_t last_dim(const shape_t &shape)
    {
        const auto rank = shape.rank();
        check(rank > 0);
        return shape.dims[rank - 1];
    }

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        const auto bias_init = gc(new constant_initializer_t(0.1));
        const auto weight_init = gc(new truncated_normal_initializer_t(0.1));
        const auto c = last_dim(x->shape);
        const auto weight =
            ctx.make_parameter(shape_t(r, s, c, d), weight_init);
        auto y = ctx.make_operator(*op_conv_nhwc, x, weight);
        const auto bias = ctx.make_parameter(shape_t(d), bias_init);
        y = ctx.make_operator(*op_add, y, bias);
        return y;
    }
};

s_layer_t *const new_layer_conv_nhwc(uint32_t r, uint32_t s, uint32_t d)
{
    return new conv_nhwc(r, s, d);
}
