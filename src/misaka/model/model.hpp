#pragma once
#include <string>

#include <misaka.h>
#include <misaka/core/gc.hpp>
#include <misaka/core/shape.hpp>
#include <misaka/model/node.hpp>

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

    GC<node_t> gc;
    std::vector<operator_node_t *> ops;
    std::vector<parameter_node_t *> params;

    node_t *make_parameter(const shape_t &shape, std::string name = "")
    {
        if (name.empty()) {
            name = param_name();
        }
        auto p = new parameter_node_t(shape, name);
        params.push_back(p);
        return gc(p);
    }

    node_t *make_placeholder(const shape_t &shape, std::string name = "")
    {
        if (name.empty()) {
            name = place_name();
        }
        return gc(new placeholder_node_t(shape, name));
    }

    node_t *make_operator(const operator_t &op, node_t *nodes[],
                          std::string name = "")
    {
        if (name.empty()) {
            name = op_name();
        }
        auto o = new operator_node_t(op, nodes, name);
        ops.push_back(o);
        return gc(o);
    }

    node_t *wrap(const shape_t &shape, const node_t &node)
    {
        return gc(new wrap_node_t(shape, node));
    }

    void debug() const
    {
        using T = float;
        printf("debug:\n");
        for (auto p : params) {
            r_tensor_ref_t<T> r(p->value());
            r_tensor_ref_t<T> s(p->gradient());
            printf("%-16s: ", p->name.c_str());
            print(r);
            printf("%-16s: ", p->name.c_str());
            print(s);
        }
        for (auto o : ops) {
            r_tensor_ref_t<T> r(o->value());
            r_tensor_ref_t<T> s(o->gradient());
            printf("%-16s: ", o->name.c_str());
            print(r);
            printf("%-16s: ", o->name.c_str());
            print(s);
        }
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
