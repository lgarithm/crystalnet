#pragma once
#include <cstdint>
#include <tuple>
#include <vector>

#include <misaka/core/shape.hpp>
#include <misaka/linag/base.hpp>
#include <misaka/linag/linag.hpp>
#include <teavana/range.hpp>

template <typename T>
matrix_ref_t<T> cast_to_m(uint32_t m, uint32_t n,
                          const r_tensor_ref_t<T> &tensor)
{
    assert(m * n == tensor.shape.dim());
    return matrix_ref_t<T>(m, n, tensor.data);
}

using tea::range;

struct conv_nhwc {
    constexpr static uint8_t arity = 2;

    // [n, h, w, c], [r, s, c, d] -> [n, u, v, d]
    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        const auto[n, h, w, c] = cast<4>((*shape_list)[0].dims);
        const auto[r, s, _c, d] = cast<4>((*shape_list)[1].dims);
        assert(c == _c);
        return new shape_t(n, h - r + 1, w - s + 1, d);
    }

    using T = float; // TODO: cast based on dtype

    static void lowering(const r_tensor_ref_t<T> &x,
                         const r_tensor_ref_t<T> &x_)
    {
        const auto[n, h, w, c] = cast<4>(x.shape.dims);
        const auto[_n, u, v, r, s, _c] = cast<6>(x_.shape.dims);
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
        const auto[n, h, w, c] = cast<4>(x.shape.dims);
        const auto[_n, u, v, r, s, _c] = cast<6>(x_.shape.dims);
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
            DEBUG(__FILE__);
            const auto x = r_tensor_ref_t<T>(inputs[0]);
            const auto y = r_tensor_ref_t<T>(inputs[1]);
            const auto z = r_tensor_ref_t<T>(output);

            const auto[n, h, w, c] = cast<4>(x.shape.dims);
            const auto[r, s, _c, d] = cast<4>(y.shape.dims);
            const auto[_n, u, v, _d] = cast<4>(z.shape.dims);
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

            const auto[n, h, w, c] = cast<4>(x.shape.dims);
            const auto[r, s, _c, d] = cast<4>(y.shape.dims);
            const auto[_n, u, v, _d] = cast<4>(z.shape.dims);

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

operator_t *op_conv_nhwc = _register_bi_op<conv_nhwc>("conv_nhwc");
