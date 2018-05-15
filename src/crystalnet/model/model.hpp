#pragma once
#include <string>

#include <crystalnet.h>
#include <crystalnet/core/context.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/model/node.hpp>
#include <crystalnet/model/parameter.hpp>

struct model_ctx_t : named_context_t<node_t> {

    parameter_ctx_t &p_ctx;

    Ref<operator_node_t> ops;
    Ref<parameter_node_t> params;
    Ref<placeholder_node_t> places;

    explicit model_ctx_t(parameter_ctx_t &p_ctx)
        : named_context_t<node_t>("node"), p_ctx(p_ctx)
    {
    }

    node_t *make_parameter(const std::string &name, const shape_t &shape)
    {
        auto t = p_ctx.make_parameter(name, shape);
        return own(params(new parameter_node_t(name, t)), name);
    }

    node_t *make_placeholder(const std::string &name, const shape_t &shape)
    {
        return own(places(new placeholder_node_t(name, shape)), name);
    }

    node_t *make_operator(const std::string &name, const operator_t &op,
                          const node_t *nodes[])
    {
        return own(ops(new operator_node_t(name, op, nodes)), name);
    }

    node_t *wrap(const std::string &name, const shape_t &shape,
                 const node_t &node)
    {
        return own(new wrap_node_t(name, shape, node), name);
    }
};

struct model_t {
    model_ctx_t &ctx;
    node_t &input;
    const node_t &output;

    model_t(model_ctx_t &ctx, node_t &input, node_t &output)
        : ctx(ctx), input(input), output(output)
    {
    }
};
