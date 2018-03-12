#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/model/model.hpp>

inline void log_new_s_node(const shape_t &shape, const char *type)
{
    printf("[D] %s %s\n", type, std::to_string(shape).c_str());
}

struct model_option_t;

struct s_node_t {
    const shape_t shape;
    // const std::string name; // TODO: require
    s_node_t(const shape_t &shape) : shape(shape) {}

    virtual node_t *realize(model_ctx_t &, const model_option_t &) const = 0;
    virtual ~s_node_t() {}
};

struct model_option_t {
    const s_node_t *const input;
    const uint32_t batch_size;
    model_option_t(const s_node_t *input, uint32_t batch_size)
        : input(input), batch_size(batch_size)
    {
    }
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
    const initializer_t *const init;
    explicit s_parameter_node_t(const shape_t &shape,
                                const initializer_t *const init = nullptr)
        : s_node_t(shape), init(init)
    {
        log_new_s_node(shape, "covar");
    }
    node_t *realize(model_ctx_t &ctx, const model_option_t &opt) const override
    {
        const auto p = ctx.make_parameter(shape, "", this);
        if (init) {
            (*init)(p->value());
        }
        return p;
    }
};

struct s_placeholder_node_t : s_node_t {
    explicit s_placeholder_node_t(const shape_t &shape) : s_node_t(shape)
    {
        log_new_s_node(shape, "var");
    }
    node_t *realize(model_ctx_t &ctx, const model_option_t &opt) const override
    {
        printf("s_placeholder_node_t::realize %s\n",
               std::to_string(shape).c_str());
        if (this == opt.input) {
            return ctx.make_placeholder(shape.batch(opt.batch_size));
        }
        return ctx.make_placeholder(shape);
    }
};

struct s_operator_node_t : s_node_t {
    const operator_t &op;
    const s_node_list_t inputs;

    static shape_t infer(const operator_t &op, const s_node_list_t &inputs)
    {
        return (*op.infer)(shape_list_t(inputs.shapes()));
    }

    s_operator_node_t(const operator_t &op, const s_node_list_t &inputs)
        : s_node_t(infer(op, inputs)), op(op), inputs(inputs)
    {
        log_new_s_node(shape, ("op::" + op.name).c_str());
    }
    node_t *realize(model_ctx_t &ctx, const model_option_t &opt) const override
    {
        std::vector<node_t *> _args;
        for (auto p : inputs.nodes) {
            _args.push_back(p->realize(ctx, opt));
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
    node_t *realize(model_ctx_t &ctx, const model_option_t &opt) const override
    {
        node_t *node = wrapped->realize(ctx, opt);
        if (node->shape.dim() != shape.dim()) {
            return ctx.wrap(shape.batch(opt.batch_size), *node);
        }
        return ctx.wrap(shape, *node);
    }
};

model_t *realize(parameter_ctx_t *, const s_model_t *, uint32_t);
