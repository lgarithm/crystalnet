

#include <limits>

#include <crystalnet-contrib/yolo/pool_layer.h>
#include <crystalnet-contrib/yolo/yolo.hpp>
#include <crystalnet-internal.h>
#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/operator.hpp>  // TODO: don't include private headers
#include <crystalnet/layers/layer.hpp>
#include <crystalnet/utility/range.hpp>

namespace darknet
{
template <typename T>
void max_pool(const matrix_ref_t<T> &x, const matrix_ref_t<T> &y,  //
              uint32_t size, uint32_t stride)
{
    const auto [h, w] = y.shape.dims;
    for (const auto i_ : range(h)) {
        for (const auto j_ : range(w)) {
            T z = std::numeric_limits<T>::lowest();
            for (const auto u : range(size)) {
                for (const auto v : range(size)) {
                    const uint32_t i = i_ * stride + u;
                    const uint32_t j = j_ * stride + v;
                    z = std::max(z, x.at(i, j));
                }
            }
            y.at(i_, j_) = z;
        }
    }
}

struct max_pool_op {
    constexpr static uint8_t arity = 1;

    const uint32_t filter_size;
    const uint32_t stride;

    max_pool_op(uint32_t filter_size, uint32_t stride)
        : filter_size(filter_size), stride(stride)
    {
    }

    uint32_t output_size(uint32_t input_size, uint32_t filter_size,
                         uint32_t stride) const
    {
        check(input_size >= filter_size);
        return (input_size - filter_size) / stride + 1;
    }

    shape_t infer(const shape_list_t &shape_list) const
    {
        const auto [p] = cast<arity>(shape_list.shapes, auto_hint);
        if (p.rank() == 3) {
            return _infer(shape_list_t({p.batch(1)}));
        } else {
            return _infer(shape_list);
        }
    }

    shape_t _infer(const shape_list_t &shape_list) const
    {
        const auto [p] = cast<arity>(shape_list.shapes, auto_hint);
        const auto [b, c, h, w] = cast<4>(p.dims, auto_hint);
        const uint32_t h_ = output_size(h, filter_size, stride);
        const uint32_t w_ = output_size(w, filter_size, stride);
        return shape_t(b, c, h_, w_);
    }

    using T = float;  // TODO: cast based on dtype

    void forward(const forward_ctx_t &ctx) const
    {
        const auto [x] = cast<arity>(ctx.inputs._args, auto_hint);
        const auto [n, c, h, w] = cast<4>(x.shape.dims, auto_hint);
        const auto y = ctx.output;
        const auto [_n, _c, h_, w_] = cast<4>(y.shape.dims, auto_hint);

        for (const auto b : range(n)) {
            const auto xx = ranked<4, T>(x)[b];
            const auto yy = ranked<4, T>(y)[b];
            for (const auto l : range(c)) {
                const auto xxx = xx[l];
                const auto yyy = yy[l];
                max_pool(xxx, yyy, filter_size, stride);
            }
        }
    }

    void backward(const backward_ctx_t &ctx) const
    {
        throw std::logic_error("NOT IMPLEMENTED");
    }
};

struct max_pool_layer : s_layer_t {
    const uint32_t size;
    const uint32_t stride;
    std::unique_ptr<max_pool_op> _op;
    const operator_t *op;

    max_pool_layer(uint32_t size, uint32_t stride)
        : size(size), stride(stride), _op(new max_pool_op(size, stride)),
          op(_register_generic_bi_op("darknet::max_pool", _op.get()))
    {
    }

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        return ctx.make_operator(
            *op, {x}, "maxpool_" + std::to_string(get_layer_number(ctx)));
    }
};
}  // namespace darknet

s_layer_t *pool(uint32_t size, uint32_t stride)
{
    return new darknet::max_pool_layer(size, stride);
}
