#pragma once
#include <crystalnet.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/layers/layer.hpp>
#include <crystalnet/ops/const.hpp>
#include <crystalnet/ops/truncated_normal.hpp>
#include <crystalnet/utility/cast.hpp>

struct dense : s_layer_t {
    static GC<initializer_t> gc;
    const uint32_t n;
    explicit dense(uint32_t n) : n(n) {}

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        const auto bias_init = gc(new constant_initializer_t(0.1));
        const auto weight_init = gc(new truncated_normal_initializer_t(0.1));
        const uint32_t n0 = x->shape.dim();
        if (x->shape.rank() != 1) {
            x = ctx.wrap_node(shape_t(n0), x);
        }
        auto w = ctx.make_parameter(shape_t(n0, n), weight_init);
        auto y = ctx.make_operator(*op_mul, x, w);
        auto b = ctx.make_parameter(shape_t(n), bias_init);
        return ctx.make_operator(*op_add, y, b);
    }

    static s_layer_t *create(const shape_list_t *shape_list)
    {
        check(shape_list->size() == 1);
        const auto[n] = cast<1>((*shape_list)[0].dims);
        return new dense(n);
    }
};
