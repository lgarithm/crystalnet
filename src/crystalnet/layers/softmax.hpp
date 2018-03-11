#pragma once
#include <crystalnet.h>
#include <crystalnet/layers/layer.hpp>
#include <crystalnet/ops/softmax.hpp>

template <> struct op_instance<softmax> {
    static operator_t *get() { return op_softmax; }
};

struct softmax_layer : unary_op_layer<softmax> {
    static s_layer_t *create(const shape_list_t *shape_list)
    {
        return new softmax_layer();
    }
};
