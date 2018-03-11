#pragma once
#include <crystalnet.h>
#include <crystalnet/symbol/model.hpp>
#include <crystalnet/symbol/node.hpp>

// TODO: layer APIs for C
struct s_layer_t {
    virtual s_node_t *operator()(s_model_ctx_t &, s_node_t *x) const = 0;
    virtual ~s_layer_t() {}
};

template <typename T> struct op_instance;

template <typename T> struct unary_op_layer : s_layer_t {
    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        return ctx.make_operator(*op_instance<T>::get(), x);
    }
};
