#pragma once
#include <cstdint>
#include <string>
#include <type_traits>

#include <crystalnet.h>
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
    tensor_ref_list_t input_gradients; // TODO: make items of it optional
    tensor_ref_t output_gradient;

    backward_ctx_t(const tensor_ref_list_t &inputs, const tensor_ref_t &output,
                   const tensor_ref_list_t &input_gradients,
                   const tensor_ref_t &output_gradient)
        : inputs(inputs), output(output), input_gradients(input_gradients),
          output_gradient(output_gradient)
    {
    }
};

struct operator_t {
    const uint8_t arity;
    operator_t(const char *const name, uint8_t arity, shape_func_t infer,
               forward_func_t eval, backward_func_t feed)
        : name(name), arity(arity), infer(infer), forward(eval), backward(feed)
    {
    }

    const std::string name;

    shape_func_t *infer;
    forward_func_t *forward;
    backward_func_t *backward;
};

template <typename T> struct operator_creator_t {
    static void forward(forward_ctx_t *ctx) { (*(typename T::forward *)ctx)(); }
    static void backward(backward_ctx_t *ctx)
    {
        (*(typename T::backward *)ctx)();
    }
};

template <typename T> operator_t *_register_bi_op(const char *const name)
{
    return register_op(name, T::arity, T::infer, operator_creator_t<T>::forward,
                       operator_creator_t<T>::backward);
}

template <typename T, typename S> void call(S &op)
{
    static_assert(std::is_base_of<S, T>::value);
    (*(T *)&op)();
}
