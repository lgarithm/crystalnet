#include <crystalnet-ext.h>
#include <crystalnet/layers/conv_nhwc.hpp>
#include <crystalnet/layers/dense.hpp>
#include <crystalnet/layers/pool.hpp>
#include <crystalnet/layers/relu.hpp>
#include <crystalnet/layers/softmax.hpp>

s_node_t *transform(s_model_ctx_t *ctx, const s_layer_t *l, s_node_t *x)
{
    return ctx->_layers((*l)(*ctx, x));
}

void del_s_layer(s_layer_t *layer) { delete layer; }

// layer constructors

s_layer_t *const new_layer_dense(uint32_t n) { return new dense(n); }

s_layer_t *const new_layer_conv_nhwc(uint32_t r, uint32_t s, uint32_t d)
{
    return new conv_nhwc(r, s, d);
}

s_layer_t *const new_layer_softmax() { return new softmax_layer; }

s_layer_t *const new_layer_relu() { return new relu_layer; }

s_layer_t *const new_layer_pool_max() { return new pool; }
