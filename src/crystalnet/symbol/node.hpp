#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/model/model.hpp>

struct model_option_t;

struct s_node_t {
    static void on_create(const shape_t &shape, const char *type)
    {
        printf("[D] %s %s\n", type, std::to_string(shape).c_str());
    }

    const shape_t shape;
    // const std::string name; // TODO: require
    s_node_t(const shape_t &shape) : shape(shape) {}

    virtual node_t *realize(model_ctx_t &, const model_option_t &) const = 0;
    virtual ~s_node_t() {}
};

struct model_option_t {
    const s_node_t *const input;
    const uint32_t batch_size;
    const bool batch = true; // TODO: support unbatched realization

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
        on_create(shape, "covar");
    }

    node_t *realize(model_ctx_t &ctx, const model_option_t &opt) const override
    {
        const auto p = ctx.make_parameter(shape, "", this);
        if (init) {
            (*init)(p->value());
        }
        printf("[D] s_parameter_node_t::realize %s -> %s\n",
               std::to_string(shape).c_str(), std::to_string(p->shape).c_str());
        return p;
    }
};

struct s_placeholder_node_t : s_node_t {
    explicit s_placeholder_node_t(const shape_t &shape) : s_node_t(shape)
    {
        on_create(shape, "var");
    }

    node_t *realize(model_ctx_t &ctx, const model_option_t &opt) const override
    {
        const auto ret = [&]() {
            if (this == opt.input) {
                return ctx.make_placeholder(shape.batch(opt.batch_size));
            }
            return ctx.make_placeholder(shape);
        }();
        printf("[D] s_placeholder_node_t::realize %s -> %s\n",
               std::to_string(shape).c_str(),
               std::to_string(ret->shape).c_str());
        return ret;
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
        on_create(shape, ("op." + op.name).c_str());
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
    const shape_t original_shape;
    const s_node_t *const wrapped;

    s_wrap_node_t(const shape_t &shape, const s_node_t *node)
        : s_node_t(shape), original_shape(node->shape), wrapped(node)
    {
        // on_create(shape, "wrap");
        printf("[D] %s %s -> %s\n", "wrap",
               std::to_string(original_shape).c_str(),
               std::to_string(shape).c_str());
    }

    node_t *realize(model_ctx_t &ctx, const model_option_t &opt) const override
    {
        const auto ret = [&]() {
            node_t *node = wrapped->realize(ctx, opt);
            if (!(node->shape == original_shape)) {
                check(node->shape == original_shape.batch(opt.batch_size));
                return ctx.wrap(shape.batch(opt.batch_size), *node);
            }
            return ctx.wrap(shape, *node);
        }();
        printf("[D] s_wrap_node_t::realize %s -> %s\n",
               std::to_string(shape).c_str(),
               std::to_string(ret->shape).c_str());
        return ret;
    }
};
