#pragma once
#include <cstdint>
#include <tuple>
#include <vector>

#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/utility/range.hpp>

template <typename T>
matrix_ref_t<T> cast_to_m(uint32_t m, uint32_t n,
                          const r_tensor_ref_t<T> &tensor)
{
    check(m * n == tensor.shape.dim());
    return matrix_ref_t<T>(ranked_shape_t<2>(m, n), tensor.data);
}

struct conv_nhwc {
    constexpr static uint8_t arity = 2;

    // [n, h, w, c], [r, s, c, d] -> [n, u, v, d]
    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list.shapes, auto_hint);
        const auto[n, h, w, c] = cast<4>(p.dims, auto_hint);
        const auto[r, s, _c, d] = cast<4>(q.dims, auto_hint);
        check(c == _c);
        return shape_t(n, h - r + 1, w - s + 1, d);
    }

    using T = float; // TODO: cast based on dtype

    static void lowering(const r_tensor_ref_t<T> &x,
                         const r_tensor_ref_t<T> &x_)
    {
        const auto[n, h, w, c] = cast<4>(x.shape.dims, auto_hint);
        const auto[_n, u, v, r, s, _c] = cast<6>(x_.shape.dims, auto_hint);
        T *px_ = x_.data;
        for (auto l : range(n)) {
            for (auto i : range(u)) {
                for (auto j : range(v)) {
                    for (auto p : range(r)) {
                        for (auto q : range(s)) {
                            for (auto k : range(c)) {
                                // *px_++ = x.at(l, p + i, q + j, k);
                                // x   ::   [n, h,     w,     c]
                                // *px_++ = [l, p + i, q + j, k]
                                const auto idx =
                                    ((l * h + p + i) * w + q + j) * c + k;
                                *px_++ = x.data[idx];
                            }
                        }
                    }
                }
            }
        }
    }

    static void uppering(const r_tensor_ref_t<T> &x,
                         const r_tensor_ref_t<T> &x_)
    {
        const auto[n, h, w, c] = cast<4>(x.shape.dims, auto_hint);
        const auto[_n, u, v, r, s, _c] = cast<6>(x_.shape.dims, auto_hint);
        x.fill(0);
        T *px_ = x_.data;
        for (auto l : range(n)) {
            for (auto i : range(u)) {
                for (auto j : range(v)) {
                    for (auto p : range(r)) {
                        for (auto q : range(s)) {
                            for (auto k : range(c)) {
                                const auto idx =
                                    ((l * h + p + i) * w + q + j) * c + k;
                                x.data[idx] += *px_++;
                            }
                        }
                    }
                }
            }
        }
    }

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto x = r_tensor_ref_t<T>(inputs[0]);
            const auto y = r_tensor_ref_t<T>(inputs[1]);
            const auto z = r_tensor_ref_t<T>(output);

            const auto[n, h, w, c] = cast<4>(x.shape.dims, auto_hint);
            const auto[r, s, _c, d] = cast<4>(y.shape.dims, auto_hint);
            const auto[_n, u, v, _d] = cast<4>(z.shape.dims, auto_hint);
            // TODO: use mpool

            tensor_t x__(shape_t(n, u, v, r, s, c), idx_type<T>::type);
            const auto x_ = r_tensor_ref_t<T>(x__);
            lowering(x, x_);

            linag<T>::mm(cast_to_m<T>(n * u * v, r * s * c, x_),
                         cast_to_m<T>(r * s * c, d, y),
                         cast_to_m<T>(n * u * v, d, z));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto x = r_tensor_ref_t<T>(inputs[0]);
            const auto y = r_tensor_ref_t<T>(inputs[1]);
            const auto z = r_tensor_ref_t<T>(output);
            const auto gx = r_tensor_ref_t<T>(input_gradients[0]);
            const auto gy = r_tensor_ref_t<T>(input_gradients[1]);
            const auto gz = r_tensor_ref_t<T>(output_gradient);

            const auto[n, h, w, c] = cast<4>(x.shape.dims, auto_hint);
            const auto[r, s, _c, d] = cast<4>(y.shape.dims, auto_hint);
            const auto[_n, u, v, _d] = cast<4>(z.shape.dims, auto_hint);

            tensor_t gx__(shape_t(n, u, v, r, s, c), idx_type<T>::type);
            const auto gx_ = r_tensor_ref_t<T>(gx__);

            linag<T>::mmt(cast_to_m<T>(n * u * v, d, gz),
                          cast_to_m<T>(r * s * c, d, y),
                          cast_to_m<T>(n * u * v, r * s * c, gx_));
            uppering(gx, gx_);

            tensor_t x__(shape_t(n, u, v, r, s, c), idx_type<T>::type);
            const auto x_ = r_tensor_ref_t<T>(x__);
            lowering(x, x_);
            linag<T>::mtm(cast_to_m<T>(n * u * v, r * s * c, x_),
                          cast_to_m<T>(n * u * v, d, gz),
                          cast_to_m<T>(r * s * c, d, gy));
        }
    };
};

struct conv_hwc {
    constexpr static uint8_t arity = 2;

    // [h, w, c], [r, s, c, d] -> [h - r + 1, w - s + 1, d]
    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list.shapes, auto_hint);
        const auto[h, w, c] = cast<3>(p.dims, auto_hint);
        const auto[r, s, _c, d] = cast<4>(q.dims, auto_hint);
        check(c == _c);
        return shape_t(h - r + 1, w - s + 1, d);
    }

    using T = float; // TODO: cast based on dtype

    static tensor_ref_t as_one_batch(const tensor_ref_t &r)
    {
        return tensor_ref_t(r.shape.batch(1), r.dtype, r.data);
    }

    struct forward : forward_ctx_t {
        void operator()() const
        {
            forward_ctx_t ctx(
                tensor_ref_list_t({as_one_batch(inputs[0]), inputs[1]}),
                as_one_batch(output));
            call<conv_nhwc::forward>(ctx);
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            backward_ctx_t ctx(
                tensor_ref_list_t({as_one_batch(inputs[0]), inputs[1]}),
                as_one_batch(output),
                tensor_ref_list_t(
                    {as_one_batch(input_gradients[0]), input_gradients[1]}),
                as_one_batch(output_gradient));
            call<conv_nhwc::backward>(ctx);
        }
    };
};

struct conv2d {
    constexpr static uint8_t arity = 2;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list.shapes, auto_hint);
        if (p.rank() == 3) {
            return conv_hwc::infer(shape_list);
        } else {
            check(p.rank() == 4);
            return conv_nhwc::infer(shape_list);
        }
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<2>(inputs.shapes().shapes, auto_hint);
            if (p.rank() == 3) {
                forward_ctx_t ctx(*this);
                call<conv_hwc::forward>(ctx);
            } else {
                check(p.rank() == 4);
                forward_ctx_t ctx(*this);
                call<conv_nhwc::forward>(ctx);
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<2>(inputs.shapes().shapes, auto_hint);
            if (p.rank() == 3) {
                backward_ctx_t ctx(*this);
                call<conv_hwc::backward>(ctx);
            } else {
                check(p.rank() == 4);
                backward_ctx_t ctx(*this);
                call<conv_nhwc::backward>(ctx);
            }
        }
    };
};
