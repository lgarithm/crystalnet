#pragma once
#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/ops/batch.hpp>
#include <crystalnet/utility/cast.hpp>
#include <crystalnet/utility/range.hpp>

struct pool2d_c_max {
    constexpr static uint8_t arity = 1;

    // TODO: support customized filter size
    constexpr static uint32_t r = 2;
    constexpr static uint32_t s = 2;

    // [w, h, c] -> [w', h', c]
    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p] = cast<arity>(shape_list.shapes);
        const auto[h, w, c] = cast<3>(p.dims);
        check(h % r == 0);
        check(w % s == 0);
        return shape_t(h / r, w / s, c);
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

struct pool2d_n_c_max {
    constexpr static uint8_t arity = 1;
    using pool2d_c_max_batched = batch<pool2d_c_max, 0>;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p] = cast<arity>(shape_list.shapes);
        if (p.rank() == 3) {
            return pool2d_c_max::infer(shape_list);
        } else {
            return pool2d_c_max_batched::infer(shape_list);
        }
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto[p] = cast<arity>(inputs.shapes().shapes);
            if (p.rank() == 3) {
                forward_ctx_t ctx(*this);
                call<pool2d_c_max::forward>(ctx);
            } else {
                check(p.rank() == 4);
                forward_ctx_t ctx(*this);
                call<pool2d_c_max_batched::forward>(ctx);
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto[p] = cast<arity>(inputs.shapes().shapes);
            if (p.rank() == 3) {
                backward_ctx_t ctx(*this);
                call<pool2d_c_max::backward>(ctx);
            } else {
                check(p.rank() == 4);
                backward_ctx_t ctx(*this);
                call<pool2d_c_max_batched::backward>(ctx);
            }
        }
    };
};
