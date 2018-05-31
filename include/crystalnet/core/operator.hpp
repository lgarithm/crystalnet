#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

#include <crystalnet-internal.h>
#include <crystalnet/core/context.hpp>
#include <crystalnet/core/tensor.hpp>

struct forward_ctx_t {
    tensor_ref_list_t inputs;
    tensor_ref_t output;

    forward_ctx_t(const tensor_ref_list_t &inputs, const tensor_ref_t &output)
        : inputs(inputs), output(output)
    {
    }
};

struct backward_ctx_t {
    tensor_ref_list_t inputs;
    tensor_ref_t output;
    tensor_ref_list_t input_gradients;  // TODO: make items of it optional
    tensor_ref_t output_gradient;

    backward_ctx_t(const tensor_ref_list_t &inputs, const tensor_ref_t &output,
                   const tensor_ref_list_t &input_gradients,
                   const tensor_ref_t &output_gradient)
        : inputs(inputs), output(output), input_gradients(input_gradients),
          output_gradient(output_gradient)
    {
    }
};

struct shape_func_t {
    virtual shape_t operator()(const shape_list_t &) = 0;
    virtual ~shape_func_t() {}
};

struct forward_func_t {
    virtual void operator()(const forward_ctx_t &) = 0;
    virtual ~forward_func_t() {}
};

struct backward_func_t {
    virtual void operator()(const backward_ctx_t &) = 0;
    virtual ~backward_func_t() {}
};

struct operator_t {
    const uint8_t arity;
    operator_t(const char *const name, uint8_t arity, shape_func_t *infer,
               forward_func_t *eval, backward_func_t *feed)
        : name(name), arity(arity), infer(infer), forward(eval), backward(feed)
    {
    }

    const std::string name;

    std::unique_ptr<shape_func_t> infer;
    std::unique_ptr<forward_func_t> forward;
    std::unique_ptr<backward_func_t> backward;
};

struct initializer_t {
    virtual void operator()(const tensor_ref_t &) const = 0;
    virtual ~initializer_t() {}
};

struct simple_shape_func_t : shape_func_t {
    typedef shape_t(shape_fn_t)(const shape_list_t &);
    shape_fn_t *fn;
    explicit simple_shape_func_t(shape_fn_t *fn) : fn(fn) {}
    shape_t operator()(const shape_list_t &shape_list) override
    {
        return fn(shape_list);
    }
};

template <typename T, typename S> void call(const S &ctx)
{
    static_assert(std::is_base_of<S, T>::value);
    (*(T *)&ctx)();
}

template <typename T, typename S, typename P>
void call(const S &ctx, const P &p)
{
    static_assert(std::is_base_of<S, T>::value);
    (*(T *)&ctx)(p);
}

template <typename T> struct simple_forward_func_t : forward_func_t {
    void operator()(const forward_ctx_t &ctx) override
    {
        call<typename T::forward>(ctx);
    }
};

template <typename T> struct simple_backward_func_t : backward_func_t {
    void operator()(const backward_ctx_t &ctx) override
    {
        call<typename T::backward>(ctx);
    }
};

template <typename T> const operator_t *_register_bi_op(const char *const name)
{
    return register_op(name, T::arity, new simple_shape_func_t(T::infer),
                       new simple_forward_func_t<T>,
                       new simple_backward_func_t<T>);
}

template <typename T> struct generic_shape_func_t : shape_func_t {
    const T &op;

    explicit generic_shape_func_t(const T &op) : op(op) {}

    shape_t operator()(const shape_list_t &shape_list) override
    {
        return op.infer(shape_list);
    }
};

template <typename T> struct generic_forward_func_t : forward_func_t {
    const T &op;

    explicit generic_forward_func_t(const T &op) : op(op) {}

    void operator()(const forward_ctx_t &ctx) override { op.forward(ctx); }
};

template <typename T> struct generic_backward_func_t : backward_func_t {
    const T &op;

    explicit generic_backward_func_t(const T &op) : op(op) {}

    void operator()(const backward_ctx_t &ctx) override { op.backward(ctx); }
};

struct operator_registry_t : named_context_t<operator_t> {
    operator_registry_t() : named_context_t<operator_t>("operator") {}
};

extern operator_registry_t operator_registry;

template <typename T>
const operator_t *_register_generic_bi_op(const T *op,
                                          const std::string &_name = "")
{
    const std::string name =
        _name.empty() ? operator_registry.gen_name() : _name;
    return register_op(name.c_str(), T::arity,  //
                       new generic_shape_func_t<T>(*op),
                       new generic_forward_func_t<T>(*op),
                       new generic_backward_func_t<T>(*op));
}
