#pragma once
#include <cmath>

#include <crystalnet.h>
#include <crystalnet/core/debug.hpp>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/linag/base.hpp>
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
            auto a = as_vector_ref<T>(inputs[0]);
            auto b = as_vector_ref<T>(inputs[1]);
            auto c = r_tensor_ref_t<T>(output);
            auto n = equally(a.n, b.n);
            T z = 0;
            for (auto i = 0; i < n; ++i) {
                z += a(i) * log(b(i));
            }
            c.data[0] = -z;
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const T g = r_tensor_ref_t<T>(output_gradient).data[0];
            const auto x = as_vector_ref<T>(inputs[0]);
            const auto y = as_vector_ref<T>(inputs[1]);
            const auto gx = as_vector_ref<T>(input_gradients[0]);
            const auto gy = as_vector_ref<T>(input_gradients[1]);
            const auto n = equally(x.n, y.n); // == gx.n == gy.n
            for (auto i = 0; i < n; ++i) {
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
