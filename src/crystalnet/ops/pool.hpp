#pragma once
#include <crystalnet/core/shape.hpp>
#include <crystalnet/model/operator.hpp>
#include <crystalnet/utility/range.hpp>

struct pool2d_c_max {
    constexpr static uint8_t arity = 1;

    // TODO: support customized filter size
    constexpr static uint32_t r = 2;
    constexpr static uint32_t s = 2;

    // [w, h, c] -> [w', h', c]
    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        const auto[h, w, c] = cast<3>((*shape_list)[0].dims);
        assert(h % r == 0);
        assert(w % s == 0);
        return new shape_t(h / r, w / s, c);
    }

    static uint32_t idx3(uint32_t i, uint32_t j, uint32_t k, //
                         uint32_t m, uint32_t n)
    {
        return (i * m + j) * n + k;
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            DEBUG(__FILE__);
            const auto x = r_tensor_ref_t<T>(inputs[0]);
            const auto y = r_tensor_ref_t<T>(output);
            const auto[h, w, c] = cast<3>(x.shape.dims);
            const auto[_u, v, _c] = cast<3>(y.shape.dims);
            y.fill(0);
            T *px = x.data;
            for (auto i : range(h)) {
                for (auto j : range(w)) {
                    for (auto k : range(c)) {
                        T &yy = y.data[idx3(i / r, j / s, k, v, c)];
                        yy = std::max(yy, *px++);
                    }
                }
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto gx = r_tensor_ref_t<T>(input_gradients[0]);
            const auto gy = r_tensor_ref_t<T>(output_gradient);
            const auto[h, w, c] = cast<3>(gx.shape.dims);
            const auto[u, v, _c] = cast<3>(gy.shape.dims);
            for (auto p : range(u)) {
                for (auto q : range(v)) {
                    for (auto i : range(p * r, p * r + r)) {
                        for (auto j : range(q * s, q * s + s)) {
                            for (auto k : range(c)) {
                                // TODO: only assign to max
                                gx.data[idx3(i, j, k, w, c)] =
                                    gy.data[idx3(p, q, k, v, c)];
                            }
                        }
                    }
                }
            }
        }
    };
};

operator_t *op_pool2d_c_max = _register_bi_op<pool2d_c_max>("pool2d_c_max");
