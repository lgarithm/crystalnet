#pragma once
#include <crystalnet/core/shape.hpp>
#include <crystalnet/model/layer.hpp>
#include <crystalnet/model/model.hpp>
#include <crystalnet/train/initializers.hpp>

template <typename... T> shape_t shape(T... dims)
{
    return shape_t(static_cast<uint32_t>(dims)...);
}

// auto place(model_ctx_t *ctx, const shape_t &shape)
// {
//     return make_placeholder(ctx, &shape);
// }

// auto wrap(model_ctx_t *ctx, const shape_t &shape, node_t *node)
// {
//     return ctx->wrap(shape, *node);
// }

// template <typename T>
// auto var(model_ctx_t *ctx, const shape_t &shape, const T &init)
// {
//     using R = float;
//     auto p = make_parameter(ctx, &shape);
//     init(r_tensor_ref_t<R>(p->value()));
//     return p;
// }

// template <typename... T> auto apply(model_ctx_t *ctx, operator_t *op, T... a)
// {
//     node_t *args[] = {static_cast<node_t *>(a)...};
//     return make_operator(ctx, op, args);
// }
