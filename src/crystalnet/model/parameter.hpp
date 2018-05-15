#pragma once
#include <stdexcept>

#include <crystalnet.h>
#include <crystalnet/core/context.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/core/tensor.hpp>

struct parameter_ctx_t : named_context_t<tensor_t> {

    parameter_ctx_t() : named_context_t<tensor_t>("param") {}

    tensor_ref_t make_parameter(const std::string &name, const shape_t &shape)
    {
        if (index.count(name) == 0) { own(new tensor_t(shape), name); }
        const auto p = ref(*index[name]);
        check_with_hint(p.shape == shape, auto_hint);
        return p;
    }

    // TODO: support load from file directly
    void load(const std::string &name, const tensor_ref_t &r) const
    {
        const auto pos = index.find(name);
        if (pos == index.end()) {
            throw std::invalid_argument("no parameter named: " + name);
        }
        ref(*pos->second).copy_from(r);
    }
};
