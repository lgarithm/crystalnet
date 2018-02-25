#pragma once

#include <array>   // for array
#include <cstddef> // for size_t
#include <cstdint> // for uint8_t
#include <tuple>   // for tuple
#include <utility> // for index_sequence

#include "teavana/core/tensor.hpp" // for tensor_ref

namespace tea
{
template <typename R, uint8_t o_rank, uint8_t... i_ranks> struct eval_ctx {
    static constexpr uint8_t arity = sizeof...(i_ranks);
    static constexpr uint8_t output_rank = o_rank;
    static constexpr ::std::array<uint8_t, arity> input_ranks = {i_ranks...};

    using output_t = tensor_ref<R, o_rank>;
    template <uint8_t r> using input_t = tensor_ref<R, r>;

    const output_t output;
    const ::std::tuple<input_t<i_ranks>...> inputs;

    explicit eval_ctx(const output_t &output,
                      const input_t<i_ranks> &... inputs)
        : output(output), inputs(inputs...)
    {
    }
};

template <typename R, uint8_t o_rank, uint8_t... i_ranks> struct temp_ctx {
    static constexpr uint8_t arity = sizeof...(i_ranks);
    static constexpr uint8_t output_rank = o_rank;
    static constexpr ::std::array<uint8_t, arity> input_ranks = {i_ranks...};

    using output_t = tensor_ref<R, o_rank>;
    template <uint8_t r> using input_t = tensor_ref<R, r>;

    explicit temp_ctx(const eval_ctx<R, o_rank, i_ranks...> &) {}
};

template <typename R, uint8_t o_rank, uint8_t... i_ranks> struct grad_ctx {
    static constexpr uint8_t arity = sizeof...(i_ranks);
    static constexpr uint8_t output_rank = o_rank;
    static constexpr ::std::array<uint8_t, arity> input_ranks = {i_ranks...};

    using output_t = tensor_ref<R, o_rank>;
    template <uint8_t r> using input_t = tensor_ref<R, r>;

    using g_output_t = tensor_ref<R, o_rank>;
    template <uint8_t r> using g_input_t = tensor_ref<R, r>;

    const output_t output;
    const ::std::tuple<input_t<i_ranks>...> inputs;

    const g_output_t g_output;
    const ::std::tuple<g_input_t<i_ranks>...> g_inputs;

    explicit grad_ctx(const output_t &output,
                      const input_t<i_ranks> &... inputs,
                      const g_output_t &g_output,
                      const g_input_t<i_ranks> &... g_inputs)
        : output(output), inputs(inputs...), g_output(g_output),
          g_inputs(g_inputs...)
    {
    }
};

template <uint8_t pos, typename C> auto in(const C &c)
{
    return ::std::get<pos>(c.inputs);
}

template <uint8_t pos, typename C> auto gin(const C &c)
{
    return ::std::get<pos>(c.g_inputs);
}

#ifdef ENABLE_PROFILE
struct grad_0_trait {
    static constexpr const char *name = "grad_0";
};

struct grad_1_trait {
    static constexpr const char *name = "grad_1";
};
#endif

template <typename, uint8_t> struct grad_helper;

template <typename OpImpl> struct grad_helper<OpImpl, 1> {
    template <typename G, typename T> static void grad(const G &g, const T &t)
    {
#ifdef ENABLE_PROFILE
        op_count_tracer<OpImpl, grad_0_trait> counter;
#endif
        OpImpl::grad_0(g, t);
    }
};

template <typename OpImpl> struct grad_helper<OpImpl, 2> {
    template <typename G, typename T> static void grad(const G &g, const T &t)
    {
        {
#ifdef ENABLE_PROFILE
            op_count_tracer<OpImpl, grad_0_trait> counter;
#endif
            OpImpl::grad_0(g, t);
        }
        {
#ifdef ENABLE_PROFILE
            op_count_tracer<OpImpl, grad_1_trait> counter;
#endif
            OpImpl::grad_1(g, t);
        }
    }
};

template <typename...> struct select_ctx;

template <size_t r, size_t... rs>
struct select_ctx<::std::index_sequence<r, rs...>> {
    template <typename R> using temp_ctx_type = temp_ctx<R, r, rs...>;

    template <typename R> using eval_ctx_type = eval_ctx<R, r, rs...>;

    template <typename R> using grad_ctx_type = grad_ctx<R, r, rs...>;
};
}
