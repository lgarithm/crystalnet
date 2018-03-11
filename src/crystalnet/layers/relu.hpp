#pragma once
#include <crystalnet.h>
#include <crystalnet/layers/layer.hpp>
#include <crystalnet/ops/relu.hpp>

template <> struct op_instance<relu> {
    static operator_t *get() { return op_relu; }
};

struct relu_layer : unary_op_layer<relu> {
    static s_layer_t *create(const shape_list_t *shape_list)
    {
        return new relu_layer();
    }
};
