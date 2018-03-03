#pragma once
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include <misaka.h>
#include <misaka/core/debug.hpp>
#include <misaka/core/shape.hpp>
#include <misaka/core/tensor.hpp>
#include <misaka/linag/base.hpp>
#include <misaka/model/operator.hpp>

struct node_t {
    // int idx; // TODO: support index
    std::string name;
    const shape_t shape;
    const uint8_t dtype;

    static constexpr const auto default_dtype = idx_type<float>::type;

    node_t(const shape_t &shape, const char *pname = nullptr)
        : name(create_name(pname)), shape(shape), dtype(default_dtype)
    {
        LOG_NODE_USAGE(shape, name);
    }

    virtual ~node_t() {}

    virtual void bind(const tensor_ref_t &)
    {
        // TODO: move it to placeholder
        assert(false);
    }

    virtual tensor_ref_t value() const = 0;
    virtual tensor_ref_t gradient() const = 0;

    virtual void forward() const = 0;
    virtual void backward() const = 0;

    static std::string create_name(const char *pname)
    {
        if (pname) {
            return pname;
        }
        // TODO: check unique
        return "node";
    }
};

struct parameter_node_t : node_t {
    tensor_t _value;
    tensor_t _gradient;

    parameter_node_t(const shape_t &shape, const std::string &name)
        : node_t(shape, name.c_str()), _value(shape, dtype),
          _gradient(shape, dtype)
    {
    }

    tensor_ref_t value() const override { return ref(_value); }
    tensor_ref_t gradient() const override { return ref(_gradient); }

    void forward() const override { /* noop */}

    void backward() const override { /* noop */}
};

struct placeholder_node_t : node_t {
    std::unique_ptr<tensor_ref_t> _value;
    tensor_t _gradient; // TODO: remove it

    placeholder_node_t(const shape_t &shape, const std::string &name)
        : node_t(shape, name.c_str()), _gradient(shape, dtype)
    {
    }

    void bind(const tensor_ref_t &r) override
    {
        DEBUG(__func__);
        assert(r.dtype == dtype);
        _value.reset(new tensor_ref_t(r));
    }

    tensor_ref_t value() const override { return *_value; }

    tensor_ref_t gradient() const override { return ref(_gradient); }

    void forward() const override
    {
        const auto name = std::string(__func__) + "@" + this->name;
        DEBUG(name.c_str());
    }

    void backward() const override
    {
        const auto name = std::string(__func__) + "@" + this->name;
        DEBUG(name.c_str());
    }
};

struct operator_node_t : node_t {
    static shape_t infer_shape(const operator_t &op, node_t *nodes[])
    {
        const auto name = std::string(__func__) + "@" + op.name;
        DEBUG(name.c_str());
        std::vector<shape_t> shapes;
        std::string sig;
        for (auto i = 0; i < op.arity; ++i) {
            shapes.push_back(nodes[i]->shape);
            if (sig.size() > 0) {
                sig += ", ";
            }
            sig += std::to_string(nodes[i]->shape);
        }
        printf("[D] infer shape of %s from inputs shapes: %s\n",
               op.name.c_str(), sig.c_str());
        auto out_shape = op.infer(std::make_unique<shape_list_t>(shapes).get());
        printf("[D] -> %s\n", std::to_string(*out_shape).c_str());
        return *std::unique_ptr<shape_t>(out_shape);
    }

    using input_list_t = std::vector<node_t *>;
    input_list_t inputs;

    const operator_t &op;
    tensor_t _value;
    tensor_t _gradient;

    operator_node_t(const operator_t &op, node_t *inputs[],
                    const std::string &name)
        : node_t(infer_shape(op, inputs), name.c_str()),
          inputs(input_list_t(inputs, inputs + op.arity)), op(op),
          _value(this->shape), _gradient(this->shape)
    {
    }

    tensor_ref_t value() const override { return ref(_value); }
    tensor_ref_t gradient() const override { return ref(_gradient); }

    tensor_ref_list_t _input_refs() const
    {
        std::vector<tensor_ref_t> input_refs;
        for (auto i : this->inputs) {
            input_refs.push_back(i->value());
        }
        return tensor_ref_list_t(input_refs);
    }

    tensor_ref_list_t _input_grad_refs() const
    {
        std::vector<tensor_ref_t> grad_refs;
        for (auto i : this->inputs) {
            grad_refs.push_back(i->gradient());
        }
        return tensor_ref_list_t(grad_refs);
    }

    void forward() const override
    {
        const auto name = std::string(__func__) + "@" + this->name;
        DEBUG(name.c_str());
        for (auto i : inputs) {
            i->forward();
        }
        // TODO: op.forward must be present
        if (op.forward) {
            forward_ctx_t ctx(_input_refs(), ref(_value));
            op.forward(&ctx);
        }
    }

    void backward() const override
    {
        const auto name = std::string(__func__) + "@" + this->name;
        DEBUG(name.c_str());
        // TODO: op.backward should be present
        if (op.backward) {
            tensor_ref_list_t input = _input_refs();
            tensor_ref_list_t input_grads = _input_grad_refs();
            tensor_ref_t grad(_gradient);

            backward_ctx_t ctx(_input_refs(), ref(_value), _input_grad_refs(),
                               ref(_gradient));
            op.backward(&ctx);
        }
        for (auto i : inputs) {
            i->backward();
        }
    }
};

struct wrap_node_t : node_t {
    const node_t &wrapped;

    wrap_node_t(const shape_t &shape, const node_t &node)
        : node_t(shape, "wrap"), wrapped(node)
    {
        assert(shape.dim() == node.shape.dim());
    }

    tensor_ref_t value() const override
    {
        DEBUG("wrap::value");
        auto v = wrapped.value();
        return tensor_ref_t(this->shape, v.dtype, v.data);
    }
    tensor_ref_t gradient() const override
    {
        DEBUG("wrap::gradient");
        auto v = wrapped.gradient();
        return tensor_ref_t(this->shape, v.dtype, v.data);
    }

    virtual void forward() const override
    {
        DEBUG("wrap::forward");
        wrapped.forward();
    }

    virtual void backward() const override
    {
        DEBUG("wrap::backward");
        wrapped.backward();
    }
};
