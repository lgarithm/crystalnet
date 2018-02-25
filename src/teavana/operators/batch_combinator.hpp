#pragma once

#include <cassert> // for assert
#include <cstddef> // for size_t
#include <cstdint> // for uint8_t
#include <memory>  // for make_unique, unique_ptr
#include <utility> // for index_sequence, integer_sequence
#include <vector>  // for vector

#include "teavana/core/shape.hpp"  // for shape_t (ptr only), sub, shape
#include "teavana/core/tensor.hpp" // for annihilate, assign
#include "teavana/operator.hpp"    // for in, gin, select_ctx
#include "teavana/range.hpp"       // for range

namespace std
{
template <uint8_t k, typename...> struct kth;

template <typename T, T x, T... Is>
struct kth<0, ::std::integer_sequence<T, x, Is...>> {
    static constexpr T value = x;
};

template <uint8_t k, typename T, T x, T... Is>
struct kth<k, ::std::integer_sequence<T, x, Is...>> {
    static constexpr T value =
        kth<k - 1, ::std::integer_sequence<T, Is...>>::value;
};
}

namespace tea
{
template <typename...> struct batch_sig;
template <typename...> struct xbatch_sig;
template <typename...> struct xybatch_sig;
template <typename...> struct ybatch_sig;

template <size_t r, size_t r1> struct batch_sig<::std::index_sequence<r, r1>> {
    using sig = ::std::index_sequence<r + 1, r1 + 1>;
};

template <size_t r, size_t r1, size_t r2>
struct xbatch_sig<::std::index_sequence<r, r1, r2>> {
    using sig = ::std::index_sequence<r + 1, r1 + 1, r2>;
};

template <size_t r, size_t r1, size_t r2>
struct ybatch_sig<::std::index_sequence<r, r1, r2>> {
    using sig = ::std::index_sequence<r + 1, r1, r2 + 1>;
};

template <size_t r, size_t r1, size_t r2>
struct xybatch_sig<::std::index_sequence<r, r1, r2>> {
    using sig = ::std::index_sequence<r + 1, r1 + 1, r2 + 1>;
};

template <typename Op> struct batch {
    static constexpr const char *name = Op::name;
    using signature = typename batch_sig<typename Op::signature>::sig;
    using ctx_types = select_ctx<signature>;

    static constexpr auto
    infer_shape(const shape_t<::std::kth<1, signature>::value> &s1)
    {
        return shape(len(s1)) * Op::infer_shape(sub(s1));
    }

    template <typename R> struct default_impl {
        using sub_impl = typename Op::template default_impl<R>;
        using sub_temp_ctx_t = typename sub_impl::temp_ctx_t;
        using sub_eval_ctx_t = typename sub_impl::eval_ctx_t;
        using sub_grad_ctx_t = typename sub_impl::grad_ctx_t;

        using t_p = typename ctx_types::template temp_ctx_type<R>;
        using e_p = typename ctx_types::template eval_ctx_type<R>;
        using g_p = typename ctx_types::template grad_ctx_type<R>;

        struct eval_ctx_t : e_p {
            const size_t m;
            ::std::vector<::std::unique_ptr<sub_eval_ctx_t>> sub_ctxs;

            eval_ctx_t(const typename e_p::output_t &output,
                       const typename e_p::template input_t<
                           ::std::get<0>(e_p::input_ranks)> &input_0)
                : e_p(output, input_0), m(len(output))
            {
                for (auto i : range(m)) {
                    sub_ctxs.push_back(::std::make_unique<sub_eval_ctx_t>(
                        output[i], input_0[i]));
                }
            }
        };

        struct grad_ctx_t : g_p {
            const size_t m;
            ::std::vector<::std::unique_ptr<sub_grad_ctx_t>> sub_grad_ctxs;

            grad_ctx_t(const typename g_p::output_t &output,
                       const typename g_p::template input_t<
                           ::std::get<0>(g_p::input_ranks)> &input_0,
                       const typename g_p::g_output_t &g_output,
                       const typename g_p::template g_input_t<
                           ::std::get<0>(g_p::input_ranks)> &d_input_0)
                : g_p(output, input_0, g_output, d_input_0), m(len(output))
            {
                for (auto i : range(m)) {
                    sub_grad_ctxs.push_back(::std::make_unique<sub_grad_ctx_t>(
                        output[i], input_0[i], g_output[i], d_input_0[i]));
                }
            }
        };

        struct temp_ctx_t : t_p {
            const sub_temp_ctx_t sub_ctx;
            temp_ctx_t(const eval_ctx_t &ctx)
                : t_p(ctx),
                  sub_ctx(sub_eval_ctx_t(ctx.output[0], in<0>(ctx)[0]))
            {
            }
        };

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            for (auto i : range(ctx.m)) {
                sub_impl::eval(*ctx.sub_ctxs[i], t_ctx.sub_ctx);
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            for (auto i : range(len(ctx.g_output))) {
                sub_impl::grad_0(*ctx.sub_grad_ctxs[i], t_ctx.sub_ctx);
            }
        }
    };
};

template <typename Op> struct xbatch {
    static constexpr const char *name = Op::name;
    using signature = typename xbatch_sig<typename Op::signature>::sig;
    using ctx_types = select_ctx<signature>;

    static constexpr auto
    infer_shape(const shape_t<::std::kth<1, signature>::value> &s1,
                const shape_t<::std::kth<2, signature>::value> &s2)
    {
        return shape(len(s1)) * Op::infer_shape(sub(s1), s2);
    }

    template <typename R> struct default_impl {
        using sub_impl = typename Op::template default_impl<R>;
        using sub_temp_ctx_t = typename sub_impl::temp_ctx_t;
        using sub_eval_ctx_t = typename sub_impl::eval_ctx_t;
        using sub_grad_ctx_t = typename sub_impl::grad_ctx_t;

        using t_p = typename ctx_types::template temp_ctx_type<R>;
        using e_p = typename ctx_types::template eval_ctx_type<R>;
        using g_p = typename ctx_types::template grad_ctx_type<R>;

        struct eval_ctx_t : e_p {
            const size_t m;
            ::std::vector<::std::unique_ptr<sub_eval_ctx_t>> sub_ctxs;

            eval_ctx_t(const typename e_p::output_t &output,
                       const typename e_p::template input_t<
                           ::std::get<0>(e_p::input_ranks)> &input_0,
                       const typename e_p::template input_t<
                           ::std::get<1>(e_p::input_ranks)> &input_1)
                : e_p(output, input_0, input_1), m(len(output))
            {
                for (auto i : range(m)) {
                    sub_ctxs.push_back(::std::make_unique<sub_eval_ctx_t>(
                        output[i], input_0[i], input_1));
                }
            }
        };

        struct grad_ctx_t : g_p {
            const size_t m;
            ::std::vector<::std::unique_ptr<sub_grad_ctx_t>> sub_grad_ctxs;

            grad_ctx_t(const typename g_p::output_t &output,
                       const typename g_p::template input_t<
                           ::std::get<0>(g_p::input_ranks)> &input_0,
                       const typename g_p::template input_t<
                           ::std::get<1>(g_p::input_ranks)> &input_1,
                       const typename g_p::g_output_t &g_output,
                       const typename g_p::template g_input_t<
                           ::std::get<0>(g_p::input_ranks)> &d_input_0,
                       const typename g_p::template g_input_t<
                           ::std::get<1>(g_p::input_ranks)> &d_input_1)
                : g_p(output, input_0, input_1, g_output, d_input_0, d_input_1),
                  m(len(output))
            {
                for (auto i : range(m)) {
                    sub_grad_ctxs.push_back(::std::make_unique<sub_grad_ctx_t>(
                        output[i], input_0[i], input_1, g_output[i],
                        d_input_0[i], d_input_1));
                }
            }
        };

        struct temp_ctx_t : t_p {
            const sub_temp_ctx_t sub_ctx;
            const typename g_p::template g_input_t<::std::get<1>(
                g_p::input_ranks)>::own_t tmp;
            temp_ctx_t(const eval_ctx_t &ctx)
                : t_p(ctx), sub_ctx(sub_eval_ctx_t(ctx.output[0], in<0>(ctx)[0],
                                                   in<1>(ctx))),
                  tmp(in<1>(ctx).shape)
            {
            }
        };

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            for (auto i : range(ctx.m)) {
                sub_impl::eval(*ctx.sub_ctxs[i], t_ctx.sub_ctx);
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            for (auto i : range(len(ctx.g_output))) {
                sub_impl::grad_0(*ctx.sub_grad_ctxs[i], t_ctx.sub_ctx);
            }
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            annihilate(t_ctx.tmp);
            for (auto i : range(len(ctx.g_output))) {
                sub_impl::grad_1(*ctx.sub_grad_ctxs[i], t_ctx.sub_ctx);
                t_ctx.tmp += gin<1>(ctx);
            }
            assign(gin<1>(ctx), t_ctx.tmp);
        }
    };
};

template <typename Op> struct ybatch {
    static constexpr const char *name = Op::name;
    using signature = typename ybatch_sig<typename Op::signature>::sig;
    using ctx_types = select_ctx<signature>;

    static constexpr auto
    infer_shape(const shape_t<::std::kth<1, signature>::value> &s1,
                const shape_t<::std::kth<2, signature>::value> &s2)
    {
        return shape(len(s2)) * Op::infer_shape(s1, sub(s2));
    }

    template <typename R> struct default_impl {
        using sub_impl = typename Op::template default_impl<R>;
        using sub_temp_ctx_t = typename sub_impl::temp_ctx_t;
        using sub_eval_ctx_t = typename sub_impl::eval_ctx_t;
        using sub_grad_ctx_t = typename sub_impl::grad_ctx_t;

        using t_p = typename ctx_types::template temp_ctx_type<R>;
        using e_p = typename ctx_types::template eval_ctx_type<R>;
        using g_p = typename ctx_types::template grad_ctx_type<R>;

        struct eval_ctx_t : e_p {
            const size_t m;
            ::std::vector<::std::unique_ptr<sub_eval_ctx_t>> sub_ctxs;

            eval_ctx_t(const typename e_p::output_t &output,
                       const typename e_p::template input_t<
                           ::std::get<0>(e_p::input_ranks)> &input_0,
                       const typename e_p::template input_t<
                           ::std::get<1>(e_p::input_ranks)> &input_1)
                : e_p(output, input_0, input_1), m(len(output))
            {
                for (auto i : range(m)) {
                    sub_ctxs.push_back(::std::make_unique<sub_eval_ctx_t>(
                        output[i], input_0, input_1[i]));
                }
            }
        };

        struct grad_ctx_t : g_p {
            const size_t m;
            ::std::vector<::std::unique_ptr<sub_grad_ctx_t>> sub_grad_ctxs;

            grad_ctx_t(const typename g_p::output_t &output,
                       const typename g_p::template input_t<
                           ::std::get<0>(g_p::input_ranks)> &input_0,
                       const typename g_p::template input_t<
                           ::std::get<1>(g_p::input_ranks)> &input_1,
                       const typename g_p::g_output_t &g_output,
                       const typename g_p::template g_input_t<
                           ::std::get<0>(g_p::input_ranks)> &d_input_0,
                       const typename g_p::template g_input_t<
                           ::std::get<1>(g_p::input_ranks)> &d_input_1)
                : g_p(output, input_0, input_1, g_output, d_input_0, d_input_1),
                  m(len(output))
            {
                for (auto i : range(m)) {
                    sub_grad_ctxs.push_back(::std::make_unique<sub_grad_ctx_t>(
                        output[i], input_0, input_1[i], g_output[i], d_input_0,
                        d_input_1[i]));
                }
            }
        };

        struct temp_ctx_t : t_p {
            const sub_temp_ctx_t sub_ctx;
            const typename g_p::template g_input_t<::std::get<0>(
                g_p::input_ranks)>::own_t tmp;
            temp_ctx_t(const eval_ctx_t &ctx)
                : t_p(ctx), sub_ctx(sub_eval_ctx_t(ctx.output[0], in<0>(ctx),
                                                   in<1>(ctx)[0])),
                  tmp(in<0>(ctx).shape)
            {
            }
        };

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            for (auto i : range(ctx.m)) {
                sub_impl::eval(*ctx.sub_ctxs[i], t_ctx.sub_ctx);
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            annihilate(t_ctx.tmp);
            for (auto i : range(len(ctx.g_output))) {
                sub_impl::grad_0(*ctx.sub_grad_ctxs[i], t_ctx.sub_ctx);
                t_ctx.tmp += gin<0>(ctx);
            }
            assign(gin<0>(ctx), t_ctx.tmp);
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            for (auto i : range(len(ctx.g_output))) {
                sub_impl::grad_1(*ctx.sub_grad_ctxs[i], t_ctx.sub_ctx);
            }
        }
    };
};

template <typename Op> struct xybatch {
    static constexpr const char *name = Op::name;
    using signature = typename xybatch_sig<typename Op::signature>::sig;
    using ctx_types = select_ctx<signature>;

    static constexpr auto
    infer_shape(const shape_t<::std::kth<1, signature>::value> &s1,
                const shape_t<::std::kth<2, signature>::value> &s2)
    {
        assert(len(s1) == len(s2));
        return shape(len(s1)) * Op::infer_shape(sub(s1), sub(s2));
    }

    template <typename R> struct default_impl {
        using sub_impl = typename Op::template default_impl<R>;
        using sub_temp_ctx_t = typename sub_impl::temp_ctx_t;
        using sub_eval_ctx_t = typename sub_impl::eval_ctx_t;
        using sub_grad_ctx_t = typename sub_impl::grad_ctx_t;

        using t_p = typename ctx_types::template temp_ctx_type<R>;
        using e_p = typename ctx_types::template eval_ctx_type<R>;
        using g_p = typename ctx_types::template grad_ctx_type<R>;

        struct eval_ctx_t : e_p {
            const size_t m;
            ::std::vector<::std::unique_ptr<sub_eval_ctx_t>> sub_ctxs;

            eval_ctx_t(const typename e_p::output_t &output,
                       const typename e_p::template input_t<
                           ::std::get<0>(e_p::input_ranks)> &input_0,
                       const typename e_p::template input_t<
                           ::std::get<1>(e_p::input_ranks)> &input_1)
                : e_p(output, input_0, input_1), m(len(output))
            {
                for (auto i : range(m)) {
                    sub_ctxs.push_back(::std::make_unique<sub_eval_ctx_t>(
                        output[i], input_0[i], input_1[i]));
                }
            }
        };

        struct grad_ctx_t : g_p {
            const size_t m;
            ::std::vector<::std::unique_ptr<sub_grad_ctx_t>> sub_grad_ctxs;

            grad_ctx_t(const typename g_p::output_t &output,
                       const typename g_p::template input_t<
                           ::std::get<0>(g_p::input_ranks)> &input_0,
                       const typename g_p::template input_t<
                           ::std::get<1>(g_p::input_ranks)> &input_1,
                       const typename g_p::g_output_t &g_output,
                       const typename g_p::template g_input_t<
                           ::std::get<0>(g_p::input_ranks)> &d_input_0,
                       const typename g_p::template g_input_t<
                           ::std::get<1>(g_p::input_ranks)> &d_input_1)
                : g_p(output, input_0, input_1, g_output, d_input_0, d_input_1),
                  m(len(output))
            {
                for (auto i : range(m)) {
                    sub_grad_ctxs.push_back(::std::make_unique<sub_grad_ctx_t>(
                        output[i], input_0[i], input_1[i], g_output[i],
                        d_input_0[i], d_input_1[i]));
                }
            }
        };

        struct temp_ctx_t : t_p {
            const sub_temp_ctx_t sub_ctx;
            temp_ctx_t(const eval_ctx_t &ctx)
                : t_p(ctx), sub_ctx(sub_eval_ctx_t(ctx.output[0], in<0>(ctx)[0],
                                                   in<1>(ctx)[0]))
            {
            }
        };

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            for (auto i : range(ctx.m)) {
                sub_impl::eval(*ctx.sub_ctxs[i], t_ctx.sub_ctx);
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            for (auto i : range(len(ctx.g_output))) {
                sub_impl::grad_0(*ctx.sub_grad_ctxs[i], t_ctx.sub_ctx);
            }
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            for (auto i : range(len(ctx.g_output))) {
                sub_impl::grad_1(*ctx.sub_grad_ctxs[i], t_ctx.sub_ctx);
            }
        }
    };
};
}
