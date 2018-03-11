#pragma once
#include <crystalnet.h>
#include <crystalnet/layers/layer.hpp>
#include <crystalnet/ops/pool.hpp>

template <> struct op_instance<pool2d_c_max> {
    static operator_t *get() { return op_pool2d_c_max; }
};

struct pool : unary_op_layer<pool2d_c_max> {
    static s_layer_t *create(const shape_list_t *shape_list)
    {
        return new pool;
    }
};
