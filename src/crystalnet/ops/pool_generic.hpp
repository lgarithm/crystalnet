#pragma once
#include <algorithm>
#include <limits>

#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/ops/batch.hpp>
#include <crystalnet/utility/cast.hpp>
#include <crystalnet/utility/range.hpp>

struct pool2d_c {
    constexpr static uint8_t arity = 1;

    struct trait_t {
        ranked_shape_t<2> filter;
        ranked_shape_t<2> stride;

        trait_t() : filter(r_shape(2, 2)), stride(filter) {}

        explicit trait_t(const ranked_shape_t<2> &filter)
            : filter(filter), stride(filter)
        {
        }

        trait_t(const ranked_shape_t<2> &filter,
                const ranked_shape_t<2> &stride)
            : filter(filter), stride(stride)
        {
        }
    };

    static uint32_t output_size(uint32_t input_size, uint32_t filter_size,
                                uint32_t stride)
    {
        check(input_size >= filter_size);
        check((input_size - filter_size) % stride == 0);
        return (input_size - filter_size) / stride + 1;
    }

    // [w, h, c] -> [w', h', c]
    static shape_t infer(const shape_list_t &shape_list,
                         const trait_t &t = trait_t())
    {
        const auto[p] = cast<arity>(shape_list.shapes);
        const auto[n, h, w, c] = cast<4>(p.dims);
        const auto h_ = output_size(h, t.filter.dims[0], t.stride.dims[0]);
        const auto w_ = output_size(w, t.filter.dims[1], t.stride.dims[1]);
        return shape_t(n, h_, w_, c);
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()(const trait_t &t) const
        {
            const auto x = ranked<4, T>(inputs[0]);
            const auto y = ranked<4, T>(output);
            const auto[n, h_, w_, c] = y.shape.dims;
            const auto[r, s] = t.filter.dims;
            const auto[stride_r, stride_s] = t.stride.dims;

            for (auto l : range(n)) {
                for (auto k : range(c)) {
                    for (auto i_ : range(h_)) {
                        for (auto j_ : range(w_)) {
                            T yy = std::numeric_limits<T>::min();
                            for (auto u : range(r)) {
                                for (auto v : range(s)) {
                                    yy = std::max(yy, x.at(l,                 //
                                                           i_ * stride_r + u, //
                                                           j_ * stride_s + v, //
                                                           k));
                                }
                            }
                            y.at(l, i_, j_, k) = yy;
                        }
                    }
                }
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()(const trait_t &t) const
        {
            const auto x = ranked<4, T>(inputs[0]);
            const auto y = ranked<4, T>(output);
            const auto gx = ranked<4, T>(input_gradients[0]);
            const auto gy = ranked<4, T>(output_gradient);
            const auto[n, h_, w_, c] = gy.shape.dims;
            const auto[r, s] = t.filter.dims;
            const auto[stride_r, stride_s] = t.stride.dims;
            r_tensor_ref_t<T>(output_gradient).fill(0); // gy.fill(0);

            for (auto l : range(n)) {
                for (auto i_ : range(h_)) {
                    for (auto j_ : range(w_)) {
                        for (auto k : range(c)) {
                            for (auto u : range(r)) {
                                for (auto v : range(s)) {
                                    const uint32_t i = i_ * stride_r + u;
                                    const uint32_t j = j_ * stride_s + v;
                                    if (x.at(l, i, j, k) ==
                                        y.at(l, i_, j_, k)) {
                                        gx.at(l, i, j, k) +=
                                            gy.at(l, i_, j_, k);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
};

struct op_pool2d_impl_t {
    constexpr static uint8_t arity = 1;
    const pool2d_c::trait_t t;

    explicit op_pool2d_impl_t(const pool2d_c::trait_t &t) : t(t) {}

    shape_t infer(const shape_list_t &shape_list) const
    {
        const auto[p] = cast<arity>(shape_list.shapes);
        if (p.rank() == 3) {
            return pool2d_c::infer(shape_list_t({p.batch(1)}), t).sub();
        }
        return pool2d_c::infer(shape_list, t);
    }

    void forward(const forward_ctx_t &ctx) const
    {
        const auto[p] = cast<arity>(ctx.inputs.shapes().shapes);
        if (p.rank() == 3) {
            call<pool2d_c::forward>(embed(0, ctx), t);
            return;
        }
        check(p.rank() == 4);
        call<pool2d_c::forward>(ctx, t);
    }

    void backward(const backward_ctx_t &ctx) const
    {
        const auto[p] = cast<arity>(ctx.inputs.shapes().shapes);
        if (p.rank() == 3) {
            call<pool2d_c::backward>(embed(0, ctx), t);
            return;
        }
        check(p.rank() == 4);
        call<pool2d_c::backward>(ctx, t);
    }
};
