#include <crystalnet-contrib/darknet/darknet.h>
#include <crystalnet-contrib/yolo/activation.hpp>
#include <crystalnet-contrib/yolo/batch_normalization.hpp>
#include <crystalnet-contrib/yolo/bias_layer.hpp>
#include <crystalnet-contrib/yolo/conv_layer.h>
#include <crystalnet-contrib/yolo/yolo.hpp>
#include <crystalnet-internal.h>
#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/layer.hpp>
#include <crystalnet/core/operator.hpp>  // TODO: don't include private headers
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/utility/range.hpp>

namespace darknet
{
template <typename T, uint8_t r>
matrix_ref_t<T> cast_to_m(uint32_t m, uint32_t n,
                          const ranked_tensor_ref_t<T, r> &tensor)
{
    check(m * n == tensor.shape.dim());
    return matrix_ref_t<T>(ranked_shape_t<2>(m, n), tensor.data);
}

// [N, C, H, W] \circ [D, C, R, S] -> [N, D, H', W']
struct conv_nchw_op {
    constexpr static uint8_t arity = 2;

    const uint32_t stride;
    const uint32_t padding;

    conv_nchw_op(uint32_t stride, uint32_t padding)
        : stride(stride), padding(padding)
    {
    }

    uint32_t output_size(uint32_t input_size, uint32_t filter_size,
                         uint32_t padding, uint32_t stride) const
    {
        const uint32_t full_size = input_size + 2 * padding;
        check(full_size >= filter_size);
        return (full_size - filter_size) / stride + 1;
    }

    shape_t infer(const shape_list_t &shape_list) const
    {
        const auto [p, q] = cast<arity>(shape_list.shapes, auto_hint);
        if (p.rank() == 3) {
            return _infer(shape_list_t({p.batch(1), q}));
        } else {
            return _infer(shape_list);
        }
    }

    shape_t _infer(const shape_list_t &shape_list) const
    {
        const auto [p, q] = cast<arity>(shape_list.shapes, auto_hint);
        const auto [b, c, h, w] = cast<4>(p.dims, auto_hint);
        const auto [filters, _c, filter_h, filter_w] =
            cast<4>(q.dims, auto_hint);
        check(c == _c);

        uint32_t h_ = output_size(h, filter_h, padding, stride);
        uint32_t w_ = output_size(w, filter_w, padding, stride);

        return shape_t(b, filters, h_, w_);
    }

    using T = float;  // TODO: cast based on dtype

    uint32_t index(uint32_t i_, uint32_t u) const { return i_ * stride + u; }

    // [H, W] -> [R, S, H', W']
    void im2col(const ranked_tensor_ref_t<T, 2> &x,
                const ranked_tensor_ref_t<T, 4> &y) const
    {
        // std::memset(y.data, 0, sizeof(T) * y.shape.dim());
        const auto [h, w] = x.shape.dims;
        const auto [r, s, h_, w_] = y.shape.dims;
        for (const auto u : range(r)) {
            for (const auto v : range(s)) {
                for (const auto i_ : range(h_)) {
                    for (const auto j_ : range(w_)) {
                        const uint32_t i = index(i_, u);
                        const uint32_t j = index(j_, v);
                        if (i >= padding && j >= padding) {
                            y.at(u, v, i_, j_) = x.at(i - padding, j - padding);
                        } else {
                            y.at(u, v, i_, j_) = 0;
                        }
                    }
                }
            }
        }
    }

    // [C, H, W] -> [C, R, S, H', W']
    void im2col(const ranked_tensor_ref_t<T, 3> &x,
                const ranked_tensor_ref_t<T, 5> &y) const
    {
        const auto [c, h, w] = x.shape.dims;
        const auto [_c, r, s, h_, w_] = y.shape.dims;
        im2col_cpu(x.data, c, h, w, r, stride, padding, y.data);
        // for (auto i : range(c)) { im2col(x[i], y[i]); }
    }

    void forward(const forward_ctx_t &ctx) const
    {
        const auto [x, weight] = cast<arity>(ctx.inputs._args, auto_hint);
        const auto y = ranked<4, T>(ctx.output);

        const auto [n, c, h, w] = cast<4>(x.shape.dims, auto_hint);
        const auto [d, _c, r, s] = cast<4>(weight.shape.dims, auto_hint);
        const auto [_n, _d, h_, w_] = y.shape.dims;

        const tensor_t _x_col(shape_t(c, r, s, h_, w_), dtypes.f32);
        const auto xx_ = ranked<5, T>(ref(_x_col));
        const auto ww = ranked<4, T>(weight);

        // [N, C, H, W] \circ [D, C, R, S] -> [N, D, H', W']
        /*
        [N] * [C, H, W]
        [?] X [C, H, W] X -> [D, H', W']

        [D, C, R, S] X [C, R, S, H', W'] -> [D, H', W']
        */
        for (const auto b : range(n)) {
            const auto xx = ranked<3, T>(x[b]);
            // const auto yy = ranked<3, T>(y[b]);
            const auto yy = y[b];
            im2col(xx, xx_);
            using engine = linag<T, plain_impl<T>>;
            engine::mm(cast_to_m(d, c * r * s, ww),         //
                       cast_to_m(c * r * s, h_ * w_, xx_),  //
                       cast_to_m(d, h_ * w_, yy));
        }
    }

    void backward(const backward_ctx_t &ctx) const
    {
        throw std::logic_error("NOT IMPLEMENTED");
    }
};

template <typename Activate, bool BN> struct conv_layer : s_layer_t {
    const uint32_t filters;
    const uint32_t size;
    const uint32_t stride;
    const uint32_t padding;

    const shape_t bias_shape;

    conv_layer(uint32_t filters, uint32_t size, uint32_t stride,
               uint32_t padding)
        : filters(filters), size(size), stride(stride), padding(padding),
          bias_shape(shape_t(filters))
    {
    }

    uint32_t channel_size(const shape_t &shape) const
    {
        if (shape.rank() == 3) {
            return shape.dims[0];
        } else {
            check(shape.rank() == 4);
            return shape.dims[1];
        }
    }

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        const auto weight_shape =
            shape_t(filters, channel_size(x->shape), size, size);
        const int layer_number = get_layer_number(ctx);
        const std::string prefix =
            name_prefix(ctx);  // prefix for parameter name
        const std::string suffix =
            "_" + std::to_string(layer_number);  // suffix for operator name

        const auto weight = ctx.make_parameter(weight_shape, prefix + "_W");
        const auto bias = ctx.make_parameter(bias_shape, prefix + "_b");

        const auto conv_op =
            _register_generic_bi_op(gc(new conv_nchw_op(stride, padding)));
        const auto add_bias_op = _register_generic_bi_op(gc(new add_bias()));
        const auto act_op =
            _register_generic_bi_op(gc(new pointwise_op<Activate>()));

        if (BN) {
            const auto bn_op =
                _register_generic_bi_op(gc(new op_batch_norm<float>()));
            const auto scale_bias_op =
                _register_generic_bi_op(gc(new scale_bias()));

            const auto scales =
                ctx.make_parameter(bias_shape, prefix + "_scales");
            const auto rolling_means =
                ctx.make_parameter(bias_shape, prefix + "_rolling_means");
            const auto rolling_variances =
                ctx.make_parameter(bias_shape, prefix + "_rolling_variances");

            const auto y1 =
                ctx.make_operator(*conv_op, {x, weight}, "conv" + suffix);
            const auto y2 = ctx.make_operator(
                *bn_op, {y1, rolling_means, rolling_variances},
                "batch_norm" + suffix);
            const auto y3 = ctx.make_operator(*scale_bias_op, {y2, scales},
                                              "scale_bias" + suffix);
            const auto y4 = ctx.make_operator(*add_bias_op, {y3, bias},
                                              "add_bias" + suffix);
            return ctx.make_operator(*act_op, {y4}, "act" + suffix);
        } else {
            const auto y =
                ctx.make_operator(*conv_op, {x, weight}, "conv" + suffix);
            const auto z =
                ctx.make_operator(*add_bias_op, {y, bias}, "add_bias" + suffix);
            return ctx.make_operator(*act_op, {z}, "act" + suffix);
        }
    }
};

}  // namespace darknet

s_layer_t *conv(uint32_t filters, uint32_t size, uint32_t stride,
                uint32_t padding)
{
    using T = float;
    using act = leaky_relu<T>;
    return new darknet::conv_layer<act, true>(filters, size, stride, padding);
}

s_layer_t *conv_linear_act(uint32_t filters, uint32_t size, uint32_t stride,
                           uint32_t padding)
{
    using T = float;
    using act = linear<T>;
    return new darknet::conv_layer<act, false>(filters, size, stride, padding);
}
