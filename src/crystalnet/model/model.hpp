#pragma once
#include <map>
#include <string>

#include <crystalnet.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/model/node.hpp>
#include <crystalnet/model/parameter.hpp>

struct name_generator_t {
    const std::string prefix;
    uint32_t idx;

    name_generator_t(const std::string &prefix, uint32_t init = 0)
        : prefix(prefix), idx(init)
    {
    }

    std::string operator()() { return prefix + std::to_string(idx++); }
};

struct model_ctx_t {
    name_generator_t param_name = name_generator_t("param");
    name_generator_t place_name = name_generator_t("placeholder");
    name_generator_t op_name = name_generator_t("op");

    parameter_ctx_t *const p_ctx;

    GC<node_t> gc;
    Ref<operator_node_t> ops;
    Ref<parameter_node_t> params;
    Ref<placeholder_node_t> places;

    using key_t = void const *;

    model_ctx_t() : p_ctx(new parameter_ctx_t) {}
    explicit model_ctx_t(parameter_ctx_t *p_ctx) : p_ctx(p_ctx) {}

    node_t *make_parameter(const shape_t &shape, std::string name = "",
                           const key_t key = nullptr)
    {
        if (name.empty()) {
            name = param_name();
        }
        auto t = p_ctx->make_parameter(shape, key);
        return gc(params(new parameter_node_t(t, name)));
    }

    node_t *make_placeholder(const shape_t &shape, std::string name = "")
    {
        if (name.empty()) {
            name = place_name();
        }
        return gc(places(new placeholder_node_t(shape, name)));
    }

    node_t *make_operator(const operator_t &op, node_t *nodes[],
                          std::string name = "")
    {
        if (name.empty()) {
            name = op_name();
        }
        return gc(ops(new operator_node_t(op, nodes, name)));
    }

    node_t *wrap(const shape_t &shape, const node_t &node)
    {
        return gc(new wrap_node_t(shape, node));
    }

    void print_parameters() const
    {
        printf("parameters:\n");
        using T = float;
        for (auto p : params.items) {
            r_tensor_ref_t<T> r(p->value());
            r_tensor_ref_t<T> s(p->gradient());
            printf("%-16s: ", p->name.c_str());
            print(r);
            printf("%-16s: ", p->name.c_str());
            print(s);
        }
    }

    void print_opertors() const
    {
        printf("operators:\n");
        using T = float;
        for (auto o : ops.items) {
            r_tensor_ref_t<T> r(o->value());
            r_tensor_ref_t<T> s(o->gradient());
            printf("%-16s: ", o->name.c_str());
            print(r);
            printf("%-16s: ", o->name.c_str());
            print(s);
        }
    }

    void debug() const
    {
        printf("debug:\n");
        print_parameters();
        print_opertors();
    }
};

struct model_t {
    model_ctx_t *ctx;
    node_t *input;
    node_t *output;

    model_t(model_ctx_t *ctx, node_t *input, node_t *output)
        : ctx(ctx), input(input), output(output)
    {
    }
};
