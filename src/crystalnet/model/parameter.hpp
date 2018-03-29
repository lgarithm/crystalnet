#pragma once
#include <map>

#include <crystalnet.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/core/tensor.hpp>

struct parameter_ctx_t {
    GC<tensor_t> gc;
    std::map<std::string, tensor_t *> index;
    std::vector<std::pair<std::string, tensor_t *>> items;

    tensor_ref_t make_parameter(const std::string &name, const shape_t &shape)
    {
        if (index.count(name) == 0) {
            index[name] = gc(new tensor_t(shape));
            items.push_back(std::make_pair(name, index[name]));
        }
        const auto p = ref(*index[name]);
        check_with_hint(p.shape == shape, auto_hint);
        return p;
    }

    void load(const std::string &name, const tensor_ref_t &r) const
    {
        const auto pos = index.find(name);
        if (pos == index.end()) {
            printf("[W] %s: no parameter named %s\n", __func__, name.c_str());
            return;
        }
        ref(*pos->second).copy_from(r);
    }
};
