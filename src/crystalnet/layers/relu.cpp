#include <crystalnet-ext.h>
#include <crystalnet/core/layer.hpp>
#include <crystalnet/ops/relu.hpp>

struct relu_layer : s_layer_t {
    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        return ctx.make_operator(*op_relu, x);
    }
};

s_layer_t *const new_layer_relu() { return new relu_layer; }
