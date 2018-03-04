#pragma once
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/utility/range.hpp>

template <typename T>
auto change_ith(const uint8_t pos, const std::vector<T> &items,
                const T &new_item)
{
    std::vector<T> new_items;
    for (auto i : range(items.size())) {
        new_items.push_back(i == pos ? new_item : items[i]);
    }
    return new_items;
}

forward_ctx_t unbatch(uint8_t pos, uint8_t idx, const forward_ctx_t &ctx)
{
    const auto inputs = change_ith(pos, ctx.inputs._args, ctx.inputs[pos][idx]);
    return forward_ctx_t(tensor_ref_list_t(inputs), ctx.output[idx]);
}

backward_ctx_t unbatch(uint8_t pos, uint8_t idx, const backward_ctx_t &ctx)
{
    const auto inputs = change_ith(pos, ctx.inputs._args, ctx.inputs[pos][idx]);
    const auto input_gradients = change_ith(pos, ctx.input_gradients._args,
                                            ctx.input_gradients[pos][idx]);
    return backward_ctx_t(tensor_ref_list_t(inputs), ctx.output[idx],
                          tensor_ref_list_t(input_gradients),
                          ctx.output_gradient[idx]);
}

template <typename O, uint8_t pos> struct batch {
    constexpr static uint8_t arity = O::arity;

    static shape_t n_batch(uint32_t n, const shape_t &shape)
    {
        std::vector<uint32_t> dims({n});
        dims.insert(dims.end(), shape.dims.begin(), shape.dims.end());
        return shape_t(dims);
    }

    static shape_t *infer(const shape_list_t *shapes)
    {
        static_assert(pos < arity);
        assert(shapes->size() == arity);
        const auto batched_shape = (*shapes)[pos];
        assert(batched_shape.rank() > 1);
        const auto new_shapes =
            change_ith(pos, shapes->shapes, batched_shape.sub());
        const auto out_shape = std::unique_ptr<shape_t>(
            O::infer(std::make_unique<shape_list_t>(new_shapes).get()));
        return new shape_t(n_batch(batched_shape.len(), *out_shape));
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            for (auto i : range(inputs[pos].shape.len())) {
                const auto ctx = unbatch(pos, i, *this);
                (*(typename O::forward *)&ctx)();
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            for (auto i : range(inputs[pos].shape.len())) {
                const auto ctx = unbatch(pos, i, *this);
                (*(typename O::backward *)&ctx)();
            }
        }
    };
};
