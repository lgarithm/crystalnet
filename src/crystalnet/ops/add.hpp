#pragma once
#include <crystalnet.h>
#include <crystalnet/core/debug.hpp>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/linag/linag.hpp>
#include <crystalnet/ops/batch.hpp>
#include <crystalnet/utility/cast.hpp>

// [n], [n] -> [n]
struct add_vv {
    constexpr static uint8_t arity = 2;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list->shapes);
        check(p.dim() == q.dim());
        return new shape_t(p);
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            linag<T>::vv(as_vector_ref<T>(inputs[0]),
                         as_vector_ref<T>(inputs[1]), as_vector_ref<T>(output));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            auto g = r_tensor_ref_t<T>(output_gradient);
            r_tensor_ref_t<T>(input_gradients[0]).copy(g);
            r_tensor_ref_t<T>(input_gradients[1]).copy(g);
        }
    };
};

struct add {
    constexpr static uint8_t arity = 2;
    using add_mv = batch<add_vv, 0>;

    static bool is_sub(const shape_t &p, const shape_t &q)
    {
        if (p.rank() > q.rank()) {
            return false;
        }
        return p.dims ==
               std::vector<uint32_t>(q.dims.begin() + (q.rank() - p.rank()),
                                     q.dims.end());
    }

    static shape_t *infer(const shape_list_t *shape_list)
    {
        const auto[p, q] = cast<arity>(shape_list->shapes);
        if (p.rank() > q.rank()) {
            check(is_sub(q, p));
            return new shape_t(p);
        }
        check(p.rank() == q.rank());
        return add_vv::infer(shape_list);
    }

    static tensor_ref_t ref_as(const shape_t &shape, const tensor_ref_t &r)
    {
        check(shape.dim() == r.shape.dim());
        return tensor_ref_t(shape, r.dtype, r.data);
    }

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<arity>(inputs.shapes().shapes);
            if (p.rank() > q.rank()) {
                check(is_sub(q, p));
                const shape_t r(std::vector<uint32_t>(
                    p.dims.begin(), p.dims.begin() + (p.rank() - q.rank())));
                const auto m = r.dim();
                const auto n = q.dim();
                const auto[x, y] = cast<arity>(inputs._args);
                forward_ctx_t ctx(tensor_ref_list_t({ref_as(shape_t(m, n), x),
                                                     ref_as(shape_t(n), y)}),
                                  ref_as(shape_t(m, n), output));
                call<add_mv::forward>(ctx);
            } else {
                check(p.rank() == q.rank());
                forward_ctx_t ctx(*this);
                call<add_vv::forward>(ctx);
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            const auto[p, q] = cast<arity>(inputs.shapes().shapes);
            if (p.rank() > q.rank()) {
                check(is_sub(q, p));
                const shape_t r(std::vector<uint32_t>(
                    p.dims.begin(), p.dims.begin() + (p.rank() - q.rank())));
                const auto m = r.dim();
                const auto n = q.dim();
                const auto[x, y] = cast<arity>(inputs._args);
                const auto[gx, gy] = cast<arity>(input_gradients._args);
                backward_ctx_t ctx(tensor_ref_list_t({ref_as(shape_t(m, n), x),
                                                      ref_as(shape_t(n), y)}),
                                   ref_as(shape_t(m, n), output),
                                   tensor_ref_list_t({ref_as(shape_t(m, n), gx),
                                                      ref_as(shape_t(n), gy)}),
                                   ref_as(shape_t(m, n), output_gradient));
                call<add_mv::backward>(ctx);
            } else {
                check(p.rank() == q.rank());
                backward_ctx_t ctx(*this);
                call<add_vv::backward>(ctx);
            }
        }
    };
};
