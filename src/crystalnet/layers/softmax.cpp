#include <crystalnet-ext.h>
#include <crystalnet/core/layer.hpp>
#include <crystalnet/ops/softmax.hpp>

struct softmax_layer : s_layer_t {
    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        return ctx.make_operator(*op_softmax, x);
    }
};

s_layer_t *const new_layer_softmax() { return new softmax_layer; }
