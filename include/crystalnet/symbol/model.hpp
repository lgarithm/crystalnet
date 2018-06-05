#pragma once
#include <crystalnet/core/context.hpp>
#include <crystalnet/symbol/node.hpp>

struct s_model_ctx_t : named_context_t<s_node_t> {
    Ref<s_operator_node_t> ops;
    Ref<s_parameter_node_t> params;
    Ref<s_placeholder_node_t> places;
    Ref<s_node_t> _layers;

    s_model_ctx_t() : named_context_t<s_node_t>("sym") {}

    s_node_t *make_parameter(const shape_t &shape, const std::string &name)
    {
        return own(params(new s_parameter_node_t(name, shape, nullptr)), name);
    }

    s_node_t *make_parameter(const shape_t &shape,
                             const initializer_t *init = nullptr)
    {
        const auto name = gen_name("param");
        return own(params(new s_parameter_node_t(name, shape, init)), name);
    }

    s_node_t *make_placeholder(const shape_t &shape)
    {
        const auto name = gen_name("place");
        return own(places(new s_placeholder_node_t(name, shape)), name);
    }

    template <typename... T>
    s_node_t *make_operator(const operator_t &op, T... node)
    {
        return make_operator(op, s_node_list_t(node...));
    }

    s_node_t *make_operator(const operator_t &op, const s_node_list_t &args,
                            const std::string &_name = "")
    {
        const std::string name = _name.empty() ? gen_name(op.name) : _name;
        return own(ops(new s_operator_node_t(name, op, args)), name);
    }

    s_node_t *wrap_node(const shape_t &shape, const s_node_t *node)
    {
        const auto name = gen_name("wrap");
        return own(new s_wrap_node_t(name, shape, node), name);
    }
};

struct s_model_t {
    const context_t &ctx;
    const s_node_t &input;
    const s_node_t &output;

    s_model_t(const context_t &ctx, s_node_t &input, s_node_t &output)
        : ctx(ctx), input(input), output(output)
    {
    }
};
