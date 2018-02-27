#pragma once
#include <misaka.h>
#include <misaka/misaka>
#include <misaka/model/model.hpp>
#include <misaka/train/initializers.hpp>

// TODO: layer APIs for C
struct layer_t {
    using T = float;
    virtual node_t *operator()(model_ctx_t &, node_t *x) const = 0;
    virtual ~layer_t() {}
};

struct fc_layer_t : layer_t {
    const uint32_t n;
    fc_layer_t(uint32_t n) : n(n) {}

    node_t *operator()(model_ctx_t &ctx, node_t *x) const override
    {
        const truncated_normal_initializer<T> weight_init(0.1);
        const constant_initializer<T> bias_init(0.1);

        uint32_t n0 = x->shape.dim();
        if (x->shape.rank() != 1) {
            x = ctx.wrap(shape_t(n0), *x);
        }

        auto w = ctx.make_parameter(shape_t(n0, n));
        weight_init(r_tensor_ref_t<T>(w->value()));
        node_t *args1[] = {x, w};
        auto y = ctx.make_operator(*op_mul, args1);

        auto b = ctx.make_parameter(shape_t(n));
        bias_init(r_tensor_ref_t<T>(b->value()));
        node_t *args2[] = {y, b};
        return ctx.make_operator(*op_add, args2);
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

struct chain_layer_t : layer_t {
    std::vector<layer_t *> layers;

    template <typename... T>
    chain_layer_t(T... l) : layers({static_cast<layer_t *>(l)...})
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