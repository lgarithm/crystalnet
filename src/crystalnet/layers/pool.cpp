#include <crystalnet-ext.h>
#include <crystalnet/core/layer.hpp>
#include <crystalnet/ops/pool.hpp>

struct pool : s_layer_t {
    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        return ctx.make_operator(*op_pool2d_c_max, x);
    }
};

s_layer_t *const new_layer_pool_max() { return new pool; }
