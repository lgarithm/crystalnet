#pragma once
#include <crystalnet.h>
#include <crystalnet/crystalnet>
#include <crystalnet/model/model.hpp>
#include <crystalnet/train/initializers.hpp>
#include <crystalnet/utility/cast.hpp>

// TODO: layer APIs for C
struct layer_t {
    using T = float;
    virtual node_t *operator()(model_ctx_t &, node_t *x) const = 0;
    virtual ~layer_t() {}
};

using T = layer_t::T;
const truncated_normal_initializer<T> default_weight_init(0.1);
const constant_initializer<T> default_bias_init(0.1);

struct fc_layer_t : layer_t {
    const uint32_t n;
    explicit fc_layer_t(uint32_t n) : n(n) {}

    node_t *operator()(model_ctx_t &ctx, node_t *x) const override
    {
        uint32_t n0 = x->shape.dim();
        if (x->shape.rank() != 1) {
            x = ctx.wrap(shape_t(n0), *x);
        }

        auto w = ctx.make_parameter(shape_t(n0, n));
        default_weight_init(r_tensor_ref_t<T>(w->value()));
        node_t *args1[] = {x, w};
        auto y = ctx.make_operator(*op_mul, args1);

        auto b = ctx.make_parameter(shape_t(n));
        default_bias_init(r_tensor_ref_t<T>(b->value()));
        node_t *args2[] = {y, b};
        return ctx.make_operator(*op_add, args2);
    }
};

struct conv_layer_t : layer_t {
    const uint32_t r;
    const uint32_t s;
    const uint32_t d;
    conv_layer_t(uint32_t r, uint32_t s, uint32_t d = 1) : r(r), s(s), d(d) {}
    node_t *operator()(model_ctx_t &ctx, node_t *x) const override
    {
        const auto input_rank = x->shape.rank();
        auto c = [&]() {
            if (input_rank == 3) {
                const auto[h, w, c] = cast<3>(x->shape.dims);
                x = ctx.wrap(shape_t(1, h, w, c), *x);
            }
            check(x->shape.rank() == 4);
            const auto[n, h, w, c] = cast<4>(x->shape.dims);
            return c;
        }();
        auto weight = ctx.make_parameter(shape_t(r, s, c, d));
        default_weight_init(r_tensor_ref_t<T>(weight->value()));
        node_t *args1[] = {x, weight};
        auto y = ctx.make_operator(*op_conv_nhwc, args1);
        const auto[n, u, v, d] = cast<4>(y->shape.dims);
        y = ctx.wrap(shape_t(n * u * v, d), *y);
        auto bias = ctx.make_parameter(shape_t(d));
        node_t *args2[] = {y, bias};
        y = ctx.make_operator(*op_add, args2);
        if (input_rank == 3) {
            y = ctx.wrap(shape_t(u, v, d), *y);
        } else {
            y = ctx.wrap(shape_t(n, u, v, d), *y);
        }
        return y;
    }
};

struct pool_layer_t : layer_t {
    node_t *operator()(model_ctx_t &ctx, node_t *x) const override
    {
        node_t *args1[] = {x};
        auto y = ctx.make_operator(*op_pool2d_c_max, args1);
        return y;
    }
};

struct relu_layer_t : layer_t {
    node_t *operator()(model_ctx_t &ctx, node_t *x) const override
    {
        node_t *args[] = {x};
        return ctx.make_operator(*op_relu, args);
    }
};

struct softmax_layer_t : layer_t {
    node_t *operator()(model_ctx_t &ctx, node_t *x) const override
    {
        node_t *args[] = {x};
        return ctx.make_operator(*op_softmax, args);
    }
};

using fc_layer = fc_layer_t;
using relu_layer = relu_layer_t;
using softmax_layer = softmax_layer_t;
using conv_layer = conv_layer_t;
using pool_layer = pool_layer_t;

struct chain_layer_t : layer_t {
    std::vector<layer_t *> layers;

    template <typename... T>
    explicit chain_layer_t(T... l) : layers({static_cast<layer_t *>(l)...})
    {
    }

    node_t *operator()(model_ctx_t &ctx, node_t *x) const override
    {
        for (auto l : layers) {
            x = (*l)(ctx, x);
        }
        return x;
    }
};