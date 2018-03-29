#pragma once
#include <crystalnet/core/gc.hpp>
#include <crystalnet/symbol/node.hpp>

struct name_generator_t {
    const std::string prefix;
    uint32_t idx;

    name_generator_t(const std::string &prefix, uint32_t init = 0)
        : prefix(prefix), idx(init)
    {
    }

    std::string operator()() { return prefix + std::to_string(idx++); }
};

struct s_model_ctx_t {
    name_generator_t param_name = name_generator_t("param");
    name_generator_t place_name = name_generator_t("placeholder");
    name_generator_t op_name = name_generator_t("op");
    name_generator_t wrap_name = name_generator_t("wrap");

    GC<s_node_t> gc;
    Ref<s_operator_node_t> ops;
    Ref<s_parameter_node_t> params;
    Ref<s_placeholder_node_t> places;

    s_node_t *make_parameter(const shape_t &shape,
                             const initializer_t *init = nullptr)
    {
        return gc(params(new s_parameter_node_t(param_name(), shape, init)));
    }

    s_node_t *make_placeholder(const shape_t &shape)
    {
        return gc(places(new s_placeholder_node_t(place_name(), shape)));
    }

    template <typename... T>
    s_node_t *make_operator(const operator_t &op, T... node)
    {
        return make_operator(op, s_node_list_t(node...));
    }

    s_node_t *make_operator(const operator_t &op, const s_node_list_t &args)
    {
        return gc(ops(new s_operator_node_t(op_name(), op, args)));
    }

    s_node_t *wrap_node(const shape_t &shape, const s_node_t *node)
    {
        return gc(new s_wrap_node_t(wrap_name(), shape, node));
    }
};

struct s_model_t {
    const s_model_ctx_t &ctx;
    const s_node_t &input;
    const s_node_t &output;

    s_model_t(s_model_ctx_t &ctx, s_node_t &input, s_node_t &output)
        : ctx(ctx), input(input), output(output)
    {
    }
};
