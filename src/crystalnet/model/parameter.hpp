#pragma once
#include <map>

#include <crystalnet.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/core/tensor.hpp>

struct parameter_ctx_t {
    GC<tensor_t> gc;

    using key_t = void const *;
    std::map<key_t, tensor_t *> index;

    tensor_ref_t make_parameter(const shape_t &shape, const key_t key = nullptr)
    {
        if (!key) {
            return ref(*gc(new tensor_t(shape)));
        }
        if (index.count(key) == 0) {
            index[key] = gc(new tensor_t(shape));
        }
        // // TODO: validate same shape
        return ref(*index[key]);
    }
};
