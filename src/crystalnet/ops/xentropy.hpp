#pragma once
#include <cmath>

#include <crystalnet.h>
#include <crystalnet/core/error.hpp>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/utility/cast.hpp>
#include <crystalnet/utility/range.hpp>

struct xentropy_1d {
    constexpr static uint8_t arity = 2;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list.shapes);
        check(p.rank() == 1);
        check(q.rank() == 1);
        check(p.dim() == q.dim());
        return shape_t();
    }

    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            check(inputs.arity() == arity);
            auto a = ranked<1, T>(inputs[0]);
            auto b = ranked<1, T>(inputs[1]);
            auto c = ranked<0, T>(output);
            auto n = len(a);
            check(n == len(b));
            T z = 0;
            for (auto i : range(n)) {
                z += a.data[i] * log(b.data[i]);
            }
            c.data[0] = -z;
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const T g = r_tensor_ref_t<T>(output_gradient).data[0];
            const auto x = ranked<1, T>(inputs[0]);
            const auto y = ranked<1, T>(inputs[1]);
            const auto gx = ranked<1, T>(input_gradients[0]);
            const auto gy = ranked<1, T>(input_gradients[1]);
            auto n = len(x);
            check(n == len(y));
            for (auto i : range(n)) {
                gx.data[i] = g * -log(y.data[i]);
                gy.data[i] = g * (-x.data[i] / y.data[i]);
            }
        }
    };
};

struct xentropy_2d {
    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<2>(inputs.shapes().shapes);
            const auto[m, n] = cast<2>(p.dims);
            const auto[x, y] = cast<2>(inputs._args);
            for (auto i : range(m)) {
                forward_ctx_t ctx(tensor_ref_list_t({x[i], y[i]}), output[i]);
                (*(xentropy_1d::forward *)&ctx)();
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<2>(inputs.shapes().shapes);
            const auto[m, n] = cast<2>(p.dims);
            const auto[x, y] = cast<2>(inputs._args);
            const auto[gx, gy] = cast<2>(input_gradients._args);
            for (auto i : range(m)) {
                backward_ctx_t ctx(tensor_ref_list_t({x[i], y[i]}), output[i],
                                   tensor_ref_list_t({gx[i], gy[i]}),
                                   output_gradient[i]);
                (*(xentropy_1d::backward *)&ctx)();
            }
        }
    };
};

struct xentropy {
    constexpr static uint8_t arity = 2;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list.shapes);
        check(p.dims == q.dims);
        return shape_t(std::vector<uint32_t>(p.dims.begin(), p.dims.end() - 1));
    }

    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<arity>(inputs.shapes().shapes);
            if (p.rank() == 1) {
                (*(xentropy_1d::forward *)this)();
            } else {
                check(p.rank() == 2);
                (*(xentropy_2d::forward *)this)();
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<arity>(inputs.shapes().shapes);
            if (p.rank() == 1) {
                (*(xentropy_1d::backward *)this)();
            } else {
                check(p.rank() == 2);
                (*(xentropy_2d::backward *)this)();
            }
        }
    };
};
