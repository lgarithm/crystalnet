#pragma once
#include <crystalnet.h>
#include <crystalnet/symbol/model.hpp>
#include <crystalnet/symbol/node.hpp>

// TODO: layer APIs for C
struct s_layer_t {
    virtual s_node_t *operator()(s_model_ctx_t &, s_node_t *x) const = 0;
    virtual ~s_layer_t() {}
};
