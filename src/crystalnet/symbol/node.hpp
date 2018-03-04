#pragma once
#include <cstdio>
#include <memory>
#include <vector>

#include <crystalnet/core/shape.hpp>
#include <crystalnet/model/model.hpp>
#include <crystalnet/model/operator.hpp>

inline void log_new_s_node(const shape_t &shape, const char *type)
{
    printf("[D] %s %s\n", type, std::to_string(shape).c_str());
}

struct s_node_t {
    const shape_t shape;
    // const std::string name; // TODO: require
    s_node_t(const shape_t &shape) : shape(shape) {}

    virtual node_t *realize(model_ctx_t &) const = 0;
};

struct s_node_list_t {
    using T = std::vector<s_node_t *>;
    const T nodes;

    template <typename... T>
    explicit s_node_list_t(const T... node)
        : nodes({static_cast<s_node_t *>(node)...})
    {
    }

    explicit s_node_list_t(const T &nodes) : nodes(nodes) {}

    auto shapes() const
    {
        std::vector<shape_t> v;
        for (auto n : nodes) {
            v.push_back(n->shape);
        }
        return v;
    }
};

struct s_parameter_node_t : s_node_t {
    explicit s_parameter_node_t(const shape_t &shape) : s_node_t(shape)
    {
        log_new_s_node(shape, "covar");
    }
    node_t *realize(model_ctx_t &ctx) const override
    {
        return ctx.make_parameter(shape);
    }
};

struct s_placeholder_node_t : s_node_t {
    explicit s_placeholder_node_t(const shape_t &shape) : s_node_t(shape)
    {
        log_new_s_node(shape, "var");
    }
    node_t *realize(model_ctx_t &ctx) const override
    {
        return ctx.make_placeholder(shape);
    }
};

struct s_operator_node_t : s_node_t {
    const operator_t &op;
    const s_node_list_t inputs;
    s_operator_node_t(const operator_t &op, const s_node_list_t &inputs)
        : s_node_t(
              *op.infer(std::make_unique<shape_list_t>(inputs.shapes()).get())),
          op(op), inputs(inputs)
    {
        log_new_s_node(shape, "op");
    }
    node_t *realize(model_ctx_t &ctx) const override
    {
        std::vector<node_t *> _args;
        for (auto p : inputs.nodes) {
            _args.push_back(p->realize(ctx));
        }
        return ctx.make_operator(op, _args.data());
    }
};

struct s_wrap_node_t : s_node_t {
    const s_node_t *const wrapped;

    s_wrap_node_t(const shape_t &shape, const s_node_t *node)
        : s_node_t(shape), wrapped(node)
    {
        log_new_s_node(shape, "wrap");
    }
    node_t *realize(model_ctx_t &ctx) const override
    {
        return ctx.wrap(shape, *wrapped->realize(ctx));
    }
};
