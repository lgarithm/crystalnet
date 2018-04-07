#include <crystalnet-ext.h>
#include <crystalnet/core/error.hpp>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/ext/traits.hpp>
#include <crystalnet/layers/layer.hpp>
#include <crystalnet/ops/const.hpp>
#include <crystalnet/ops/truncated_normal.hpp>

struct pool2d_layer : s_layer_t {
    const operator_t *op;
    explicit pool2d_layer(const operator_t *op) : op(op) {}

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        return ctx.make_operator(*op, x);
    }
};

s_layer_t *const new_layer_pool2d(const filter_trait_t *p_filter,
                                  const stride_trait_t *p_stride)
{
    check(p_filter != nullptr);
    const auto filter = *p_filter;
    const auto stride = p_stride ? *p_stride : stride_trait_t(shape_t(1, 1));
    const auto[r, s] = cast<2>(filter.shape.dims, auto_hint);
    const auto[stride_r, stride_s] = cast<2>(stride.shape.dims, auto_hint);
    const auto op = make_op_pool2d(r, s, stride_r, stride_s);
    return new pool2d_layer(op);
}

struct conv2d_layer : s_layer_t {
    const filter_trait_t filter;
    const operator_t &conv;

    conv2d_layer(const operator_t *op, const filter_trait_t &filter)
        : conv(*op), filter(filter)
    {
        check(filter.shape.rank() == 3);
    }

    static uint32_t last_dim(const shape_t &shape)
    {
        const auto rank = shape.rank();
        check(rank > 0);
        return shape.dims[rank - 1];
    }

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        static GC<initializer_t> gc;
        const auto bias_init = gc(new constant_initializer_t(0.1));
        const auto weight_init = gc(new truncated_normal_initializer_t(0.1));

        const auto[r, s, d] = cast<3>(filter.shape.dims, auto_hint);
        const auto c = last_dim(x->shape);

        const auto weight = ctx.make_parameter(shape_t(r, s, c, d), //
                                               weight_init);
        auto y = ctx.make_operator(conv, x, weight);
        const auto bias = ctx.make_parameter(shape_t(d), bias_init);
        y = ctx.make_operator(*op_add, y, bias);
        return y;
    }
};

s_layer_t *const new_layer_conv2d(const filter_trait_t *p_filter,
                                  const padding_trait_t *p_padding,
                                  const stride_trait_t *p_stride)
{
    check(p_filter != nullptr);
    const auto filter = *p_filter;
    check(filter.shape.rank() == 3);            // [r,s,d]
    const auto padding = p_padding ? *p_padding //
                                   : padding_trait_t(shape_t(0, 0));
    const auto stride = p_stride ? *p_stride : stride_trait_t(shape_t(1, 1));

    const auto[padding_r, padding_s] = cast<2>(padding.shape.dims, auto_hint);
    const auto[stride_r, stride_s] = cast<2>(stride.shape.dims, auto_hint);
    const auto op = make_op_conv2d(padding_r, padding_s, stride_r, stride_s);
    return new conv2d_layer(op, filter);
}

s_node_t *transform_all(s_model_ctx_t *ctx, p_layer_t layers[], s_node_t *x)
{
    for (p_layer_t *pl = layers; *pl; ++pl) {
        x = transform(ctx, *pl, x);
    }
    return x;
}
