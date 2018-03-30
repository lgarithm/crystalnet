#pragma once
#include <memory>
#include <string>
#include <vector>

#include <crystalnet.h>
#include <crystalnet/core/error.hpp>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/core/tracer.hpp>

struct node_t {
    const std::string name;
    const uint8_t dtype; // TODO: pass dtype in constructor
    const shape_t shape;

    static constexpr const auto default_dtype = idx_type<float>::type;

    node_t(const std::string &name, const shape_t &shape)
        : name(name), dtype(default_dtype), shape(shape)
    {
        // LOG_NODE_USAGE(shape, name);
    }

    virtual ~node_t() {}

    virtual void bind(const tensor_ref_t &)
    {
        // TODO: move it to placeholder
        check(false);
    }

    virtual tensor_ref_t value() const = 0;
    virtual tensor_ref_t gradient() const = 0;

    virtual void forward() const = 0;
    virtual void backward() const = 0;
};

struct parameter_node_t : node_t {
    tensor_ref_t _value;
    tensor_t _gradient;

    parameter_node_t(const std::string &name, const tensor_ref_t &p)
        : node_t(name, p.shape), _value(p), _gradient(shape, dtype)
    {
    }

    tensor_ref_t value() const override { return _value; }
    tensor_ref_t gradient() const override { return ref(_gradient); }
    void forward() const override { /* noop */}
    void backward() const override { /* noop */}
};

struct placeholder_node_t : node_t {
    std::unique_ptr<tensor_ref_t> _value;
    tensor_t _gradient; // TODO: remove it

    placeholder_node_t(const std::string &name, const shape_t &shape)
        : node_t(name, shape), _gradient(shape, dtype)
    {
    }

    void bind(const tensor_ref_t &r) override
    {
        check(r.dtype == dtype);
        check(r.shape == shape);
        _value.reset(new tensor_ref_t(r));
    }

    tensor_ref_t value() const override { return *_value; }
    tensor_ref_t gradient() const override { return ref(_gradient); }
    void forward() const override { /* noop */}
    void backward() const override { /* noop */}
};

struct operator_node_t : node_t {
    static shape_t infer_shape(const operator_t &op, const node_t *nodes[])
    {
        std::vector<shape_t> shapes;
        for (auto i = 0; i < op.arity; ++i) {
            shapes.push_back(nodes[i]->shape);
        }
        return (*op.infer)(shape_list_t(shapes));
    }

    using input_list_t = std::vector<const node_t *>;
    const input_list_t inputs;
    const operator_t &op;
    tensor_t _value;
    tensor_t _gradient;

    operator_node_t(const std::string &name, const operator_t &op,
                    const node_t *inputs[])
        : node_t(name, infer_shape(op, inputs)),
          inputs(input_list_t(inputs, inputs + op.arity)), op(op),
          _value(this->shape), _gradient(this->shape)
    {
    }

    tensor_ref_t value() const override { return ref(_value); }
    tensor_ref_t gradient() const override { return ref(_gradient); }

    tensor_ref_list_t _input_refs() const
    {
        std::vector<tensor_ref_t> input_refs;
        std::transform(inputs.begin(), inputs.end(),
                       std::back_inserter(input_refs),
                       [](auto i) { return i->value(); });
        return tensor_ref_list_t(input_refs);
    }

    tensor_ref_list_t _input_grad_refs() const
    {
        std::vector<tensor_ref_t> grad_refs;
        std::transform(inputs.begin(), inputs.end(),
                       std::back_inserter(grad_refs),
                       [](auto i) { return i->gradient(); });
        return tensor_ref_list_t(grad_refs);
    }

    void forward() const override
    {
        for (auto i : inputs) {
            i->forward();
        }
        // TODO: op.forward must be present
        if (op.forward) {
            forward_ctx_t ctx(_input_refs(), ref(_value));
            TRACE_NAME(op.name, (*op.forward)(ctx));
        }
    }

    void backward() const override
    {
        // TODO: op.backward should be present
        if (op.backward) {
            backward_ctx_t ctx(_input_refs(), ref(_value), _input_grad_refs(),
                               ref(_gradient));
            TRACE_NAME(op.name, (*op.backward)(ctx));
        }
        for (auto i : inputs) {
            i->backward();
        }
    }
};

struct wrap_node_t : node_t {
    const node_t &wrapped;

    wrap_node_t(const std::string &name, const shape_t &shape,
                const node_t &node)
        : node_t(name, shape), wrapped(node)
    {
        check(shape.dim() == node.shape.dim());
    }

    tensor_ref_t value() const override
    {
        return wrapped.value().reshape(shape);
    }

    tensor_ref_t gradient() const override
    {
        return wrapped.gradient().reshape(shape);
    }

    void forward() const override { wrapped.forward(); }
    void backward() const override { wrapped.backward(); }
};
