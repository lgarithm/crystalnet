#include <crystalnet-ext.h>
#include <crystalnet/core/layer.hpp>

s_node_t *transform(s_model_ctx_t *ctx, const s_layer_t *l, s_node_t *x)
{
    return ctx->_layers((*l)(*ctx, x));
}

void del_s_layer(s_layer_t *layer) { delete layer; }
