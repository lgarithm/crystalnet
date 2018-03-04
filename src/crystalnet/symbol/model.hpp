#pragma once
#include <crystalnet/core/gc.hpp>
#include <crystalnet/symbol/node.hpp>

struct s_model_ctx_t {
    GC<s_node_t> gc;

    s_node_t *make_parameter(const shape_t &shape)
    {
        return gc(new s_parameter_node_t(shape));
    }

    s_node_t *make_placeholder(const shape_t &shape)
    {
        return gc(new s_placeholder_node_t(shape));
    }

    template <typename... T>
    s_node_t *make_operator(const operator_t &op, T... node)
    {
        return make_operator(op, s_node_list_t(node...));
    }

    s_node_t *make_operator(const operator_t &op, const s_node_list_t &args)
    {
        return gc(new s_operator_node_t(op, args));
    }

    s_node_t *wrap_node(const shape_t &shape, const s_node_t *node)
    {
        return gc(new s_wrap_node_t(shape, node));
    }
};

struct s_model_t {
    const s_model_ctx_t *const ctx;
    const s_node_t *const input;
    const s_node_t *const output;

    s_model_t(s_model_ctx_t *ctx, s_node_t *input, s_node_t *output)
        : ctx(ctx), input(input), output(output)
    {
    }
};
