#pragma once
#include <cstdint>
#include <tuple>
#include <vector>

#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/ops/batch.hpp>
#include <crystalnet/utility/cast.hpp>
#include <crystalnet/utility/range.hpp>

template <uint8_t p, uint8_t q, typename T>
matrix_ref_t<T> as_m(const ranked_tensor_ref_t<T, p + q> &t)
{
    const auto m = std::accumulate(t.shape.dims.begin(),     //
                                   t.shape.dims.begin() + p, //
                                   1, std::multiplies<uint32_t>());
    const auto n = std::accumulate(t.shape.dims.begin() + p, //
                                   t.shape.dims.end(),       //
                                   1, std::multiplies<uint32_t>());
    return matrix_ref_t<T>(ranked_shape_t<2>(m, n), t.data);
}

struct conv_nhwc_generic {
    constexpr static uint8_t arity = 2;

    struct trait_t {
        const ranked_shape_t<2> padding;
        const ranked_shape_t<2> stride;
        const ranked_shape_t<2> rate = r_shape(1, 1); // TODO: support customize

        trait_t() : padding(r_shape(0, 0)), stride(r_shape(1, 1)) {}

        explicit trait_t(const ranked_shape_t<2> &padding)
            : padding(padding), stride(r_shape(0, 0))
        {
        }

        trait_t(const ranked_shape_t<2> &padding,
                const ranked_shape_t<2> &stride)
            : padding(padding), stride(stride)
        {
        }
    };

    static uint32_t output_size(uint32_t input_size, uint32_t filter_size,
                                uint32_t padding, uint32_t stride,
                                uint32_t rate)
    {
        const uint32_t full = input_size + 2 * padding;
        const size_t patch_size = (filter_size - 1) * rate + 1;
        check(full >= patch_size);
        check((full - patch_size) % stride == 0);
        return (full - patch_size) / stride + 1;
    }

    // [n, h, w, c], [r, s, c, d] -> [n, h', w', d]
    static shape_t infer(const shape_list_t &shape_list,
                         const trait_t &t = trait_t())
    {
        const auto[p, q] = cast<arity>(shape_list.shapes, auto_hint);
        const auto[n, h, w, c] = cast<4>(p.dims, auto_hint);
        const auto[r, s, _c, d] = cast<4>(q.dims, auto_hint);
        check(c == _c);

        const auto[pad_h, pad_w] = t.padding.dims;
        const auto[stride_h, stride_w] = t.stride.dims;
        const auto[rate_h, rate_w] = t.rate.dims;

        const auto h_ = output_size(h, r, pad_h, stride_h, rate_h);
        const auto w_ = output_size(w, s, pad_w, stride_w, rate_w);
        return shape_t(n, h_, w_, d);
    }

    using T = float; // TODO: cast based on dtype

    static uint32_t f(const uint32_t u, const uint32_t p, //
                      const uint32_t stride, const uint32_t rate)
    {
        return u * stride + (p - 1) * rate + 1;
    }

    static bool g(const uint32_t i, const uint32_t h, const uint32_t pad)
    {
        return pad <= i && i < h + pad;
    }

    static void lowering(const trait_t &t, //
                         const ranked_tensor_ref_t<T, 4> &x,
                         const ranked_tensor_ref_t<T, 6> &x_)
    {
        const auto[_n, h, w, _c] = x.shape.dims;
        const auto[n, h_, w_, r, s, c] = x_.shape.dims;
        const auto[pad_h, pad_w] = t.padding.dims;
        const auto[stride_h, stride_w] = t.stride.dims;
        const auto[rate_h, rate_w] = t.rate.dims;

        T *px_ = x_.data;
        for (auto l : range(n)) {
            for (auto i_ : range(h_)) {
                for (auto j_ : range(w_)) {
                    for (auto u : range(r)) {
                        for (auto v : range(s)) {
                            for (auto k : range(c)) {
                                const auto i = f(i_, u, stride_h, rate_h);
                                const auto j = f(j_, v, stride_w, rate_w);
                                if (g(i, h, pad_h) && g(j, w, pad_w)) {
                                    *px_++ = x.at(l, i - pad_h, j - pad_w, k);
                                } else {
                                    *px_++ = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static void uppering(const trait_t &t, //
                         const ranked_tensor_ref_t<T, 4> &x,
                         const ranked_tensor_ref_t<T, 6> &x_)
    {
        const auto[_n, h, w, _c] = x.shape.dims;
        const auto[n, h_, w_, r, s, c] = x_.shape.dims;

        const auto[pad_h, pad_w] = t.padding.dims;
        const auto[stride_h, stride_w] = t.stride.dims;
        const auto[rate_h, rate_w] = t.rate.dims;

        x.fill(0);
        T *px_ = x_.data;
        for (auto l : range(n)) {
            for (auto i_ : range(h_)) {
                for (auto j_ : range(w_)) {
                    for (auto u : range(r)) {
                        for (auto v : range(s)) {
                            for (auto k : range(c)) {
                                const auto i = f(i_, u, stride_h, rate_h);
                                const auto j = f(j_, v, stride_w, rate_w);
                                if (g(i, h, pad_h) && g(j, w, pad_w)) {
                                    x.at(l, i - pad_h, j - pad_w, k) += *px_;
                                }
                                ++px_;
                            }
                        }
                    }
                }
            }
        }
    }

    struct forward : forward_ctx_t {
        void operator()(const trait_t &t) const
        {
            const auto x = ranked<4, T>(inputs[0]);
            const auto y = ranked<4, T>(inputs[1]);
            const auto z = ranked<4, T>(output);

            const auto[n, h, w, c] = x.shape.dims;
            const auto[r, s, _c, d] = y.shape.dims;
            const auto[_n, h_, w_, _d] = z.shape.dims;

            const tensor_t x__(shape_t(n, h_, w_, r, s, c), idx_type<T>::type);
            const auto x_ = ranked<6, T>(ref(x__));
            lowering(t, x, x_);
            linag<T>::mm(as_m<3, 3>(x_), as_m<3, 1>(y), as_m<3, 1>(z));
        }
    };

    struct backward : backward_ctx_t {
        void operator()(const trait_t &t) const
        {
            const auto x = ranked<4, T>(inputs[0]);
            const auto y = ranked<4, T>(inputs[1]);
            const auto z = ranked<4, T>(output);
            const auto gx = ranked<4, T>(input_gradients[0]);
            const auto gy = ranked<4, T>(input_gradients[1]);
            const auto gz = ranked<4, T>(output_gradient);

            const auto[n, h, w, c] = x.shape.dims;
            const auto[r, s, _c, d] = y.shape.dims;
            const auto[_n, h_, w_, _d] = z.shape.dims;

            {
                const tensor_t gx__(shape_t(n, h_, w_, r, s, c),
                                    idx_type<T>::type);
                const auto gx_ = ranked<6, T>(ref(gx__));
                linag<T>::mmt(as_m<3, 1>(gz), as_m<3, 1>(y), as_m<3, 3>(gx_));
                uppering(t, gx, gx_);
            }
            {
                tensor_t x__(shape_t(n, h_, w_, r, s, c), idx_type<T>::type);
                const auto x_ = ranked<6, T>(ref(x__));
                lowering(t, x, x_);
                linag<T>::mtm(as_m<3, 3>(x_), as_m<3, 1>(gz), as_m<3, 1>(gy));
            }
        }
    };
};

struct op_conv2d_impl_t {
    constexpr static uint8_t arity = 2;
    const conv_nhwc_generic::trait_t t;

    explicit op_conv2d_impl_t(const conv_nhwc_generic::trait_t &t) : t(t) {}

    shape_t infer(const shape_list_t &shape_list) const
    {
        const auto[p, q] = cast<arity>(shape_list.shapes);
        if (p.rank() == 3) {
            return conv_nhwc_generic::infer(shape_list_t({p.batch(1), q}), t)
                .sub();
        }
        return conv_nhwc_generic::infer(shape_list, t);
    }

    using T = float; // TODO: cast based on dtype

    void forward(const forward_ctx_t &ctx) const
    {
        const auto[p, q] = cast<arity>(ctx.inputs.shapes().shapes);
        if (p.rank() == 3) {
            call<conv_nhwc_generic::forward>(embed(0, ctx), t);
            return;
        }
        check(p.rank() == 4);
        call<conv_nhwc_generic::forward>(ctx, t);
    }

    void backward(const backward_ctx_t &ctx) const
    {
        const auto[p, q] = cast<arity>(ctx.inputs.shapes().shapes);
        if (p.rank() == 3) {
            call<conv_nhwc_generic::backward>(embed(0, ctx), t);
            return;
        }
        check(p.rank() == 4);
        call<conv_nhwc_generic::backward>(ctx, t);
    }
};
