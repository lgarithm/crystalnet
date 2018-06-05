#pragma once
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <vector>

#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/model/model.hpp>

struct model_option_t;

struct s_node_t {
    const std::string name;
    const shape_t shape;

    s_node_t(const std::string &name, const shape_t &shape)
        : name(name), shape(shape)
    {
    }
    virtual ~s_node_t() {}
    virtual const node_t *realize(model_ctx_t &,
                                  const model_option_t &) const = 0;
};

struct model_option_t {
    const std::string input;
    const uint32_t batch_size;
    // const bool batch = true; // TODO: support unbatched realization

    model_option_t(const std::string &input, uint32_t batch_size)
        : input(input), batch_size(batch_size)
    {
    }
};

struct s_node_list_t {
    using owner_t = std::vector<s_node_t *>;
    const owner_t nodes;

    template <typename... T>
    explicit s_node_list_t(const T... node)
        : nodes({static_cast<s_node_t *>(node)...})
    {
    }

    // TODO: deprecate
    explicit s_node_list_t(const owner_t &nodes) : nodes(nodes) {}

    // don't use explicit, TODO: disable lint
    s_node_list_t(const std::initializer_list<s_node_t *> &init) : nodes(init)
    {
    }

    auto shapes() const
    {
        std::vector<shape_t> v;
        for (auto n : nodes) { v.push_back(n->shape); }
        return v;
    }
};

struct s_parameter_node_t : s_node_t {
    const initializer_t *const init;

    explicit s_parameter_node_t(const std::string &name, const shape_t &shape,
                                const initializer_t *const init = nullptr)
        : s_node_t(name, shape), init(init)
    {
    }

    const node_t *realize(model_ctx_t &ctx,
                          const model_option_t &opt) const override
    {
        if (ctx.index.count(name) > 0) { return ctx.index[name]; }
        const auto p = ctx.make_parameter(name, shape);
        if (init) { (*init)(p->value()); }
        return p;
    }
};

struct s_placeholder_node_t : s_node_t {
    explicit s_placeholder_node_t(const std::string &name, const shape_t &shape)
        : s_node_t(name, shape)
    {
    }

    const node_t *realize(model_ctx_t &ctx,
                          const model_option_t &opt) const override
    {
        if (ctx.index.count(name) > 0) { return ctx.index[name]; }
        const auto result_shape = [&]() {
            if (name == opt.input) { return shape.batch(opt.batch_size); }
            return shape;
        }();
        return ctx.make_placeholder(name, result_shape);
    }
};

struct s_operator_node_t : s_node_t {
    const operator_t &op;
    const s_node_list_t inputs;

    static shape_t infer(const operator_t &op, const s_node_list_t &inputs)
    {
        return (*op.infer)(shape_list_t(inputs.shapes()));
    }

    s_operator_node_t(const std::string &name, const operator_t &op,
                      const s_node_list_t &inputs)
        : s_node_t(name, infer(op, inputs)), op(op), inputs(inputs)
    {
    }

    const node_t *realize(model_ctx_t &ctx,
                          const model_option_t &opt) const override
    {
        if (ctx.index.count(name) > 0) { return ctx.index[name]; }
        std::vector<const node_t *> _args;
        for (auto p : inputs.nodes) { _args.push_back(p->realize(ctx, opt)); }
        // TODO:
        // batched_op = auto_batch(op);
        // return ctx.make_operator(name, batched_op, _args.data());
        return ctx.make_operator(name, op, _args.data());
    }
};

struct s_wrap_node_t : s_node_t {
    const shape_t original_shape;
    const s_node_t *const wrapped;

    s_wrap_node_t(const std::string &name, const shape_t &shape,
                  const s_node_t *node)
        : s_node_t(name, shape), original_shape(node->shape), wrapped(node)
    {
    }

    const node_t *realize(model_ctx_t &ctx,
                          const model_option_t &opt) const override
    {
        if (ctx.index.count(name) > 0) { return ctx.index[name]; }
        const node_t *node = wrapped->realize(ctx, opt);
        if (!(node->shape == original_shape)) {
            check(node->shape == original_shape.batch(opt.batch_size));
            return ctx.wrap(name, shape.batch(opt.batch_size), *node);
        }
        return ctx.wrap(name, shape, *node);
    }
};
