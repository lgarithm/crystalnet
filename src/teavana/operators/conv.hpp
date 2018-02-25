#pragma once

#include <array>   // for get
#include <cassert> // for assert
#include <cstddef> // for size_t
#include <utility> // for integer_sequence

#include "teavana/core/shape.hpp"  // for cosub, shape, shape_t, operator+, etc
#include "teavana/core/tensor.hpp" // for ref_as, tensor_ref (ptr only), ref, etc
#include "teavana/matrix.hpp"      // for as_vector, mat_trans_x_mat, etc
#include "teavana/operator.hpp"    // for in, gin, select_ctx
#include "teavana/range.hpp"       // for range
#include "teavana/tracer.hpp"      // for DEF_COUNTER, SET_COUNTER

namespace tea
{
DEF_COUNTER(conv3d_grad_1_prepare);
DEF_COUNTER(conv3d_grad_1_mat_mul);

struct op_conv3d {
    static constexpr const char *name = "conv3d";
    using signature = ::std::index_sequence<3, 3, 3>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<3> &s1,
                                      const shape_t<3> &s2)
    {
        assert(s2 <= s1);
        return s1 + unit_shape<3>() - s2;
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;

        using t_p = typename ctx_types::template temp_ctx_type<R>;

        struct temp_ctx_t : t_p {
            const tensor<R, 6> _grad2;
            const matrix_ref<R> j2;

            temp_ctx_t(const eval_ctx_t &ctx)
                : t_p(ctx), _grad2(ctx.output.shape * in<1>(ctx).shape),
                  j2(ref_as(_grad2, shape(dim(ctx.output.shape),
                                          dim(in<1>(ctx).shape))))

            {
            }
        };

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);

            R *pz = ctx.output.data;
            for (auto k1 : range(dim<0>(ctx.output))) {
                for (auto k2 : range(dim<1>(ctx.output))) {
                    for (auto k3 : range(dim<2>(ctx.output))) {
                        R tmp = 0;
                        R *py = y.data;
                        for (auto i1 : range(k1, k1 + dim<0>(y))) {
                            for (auto i2 : range(k2, k2 + dim<1>(y))) {
                                for (auto i3 : range(k3, k3 + dim<2>(y))) {
                                    tmp += x.at(i1, i2, i3) * *py++;
                                }
                            }
                        }
                        *pz++ = tmp;
                    }
                }
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto y = in<1>(ctx);
            const auto gx = gin<0>(ctx);
            annihilate(gx);
            R *pz = ctx.g_output.data;
            for (auto k1 : range(dim<0>(ctx.g_output))) {
                for (auto k2 : range(dim<1>(ctx.g_output))) {
                    for (auto k3 : range(dim<2>(ctx.g_output))) {
                        R *py = y.data;
                        for (auto i1 : range(k1, k1 + dim<0>(y))) {
                            for (auto i2 : range(k2, k2 + dim<1>(y))) {
                                for (auto i3 : range(k3, k3 + dim<2>(y))) {
                                    gx.at(i1, i2, i3) += *pz * *py++;
                                }
                            }
                        }
                        ++pz;
                    }
                }
            }
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);
            {
                SET_COUNTER(conv3d_grad_1_prepare);
                R *p = t_ctx.j2.data;
                for (auto k1 : range(dim<0>(ctx.output))) {
                    for (auto k2 : range(dim<1>(ctx.output))) {
                        for (auto k3 : range(dim<2>(ctx.output))) {
                            for (auto i1 : range(k1, k1 + dim<0>(y))) {
                                for (auto i2 : range(k2, k2 + dim<1>(y))) {
                                    for (auto i3 : range(k3, k3 + dim<2>(y))) {
                                        *p++ = x.at(i1, i2, i3);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            {
                SET_COUNTER(conv3d_grad_1_mat_mul);
                vec_x_mat(as_vector(ctx.g_output), t_ctx.j2,
                          as_vector(gin<1>(ctx)));
            }
        }
    };
};

DEF_COUNTER(op_conv_nhwc_lowering);
DEF_COUNTER(op_conv_nhwc_uppering);

struct op_conv_nhwc {
    static constexpr const char *name = "conv_nhwc";
    using signature = ::std::index_sequence<4, 4, 4>;
    using ctx_types = select_ctx<signature>;

    static auto infer_shape(const shape_t<4> &s1, const shape_t<4> &s2)
    {
        const auto n = ::std::get<0>(s1.dims);
        const auto h = ::std::get<1>(s1.dims);
        const auto w = ::std::get<2>(s1.dims);
        const auto c = ::std::get<3>(s1.dims);
        const auto r = ::std::get<0>(s2.dims);
        const auto s = ::std::get<1>(s2.dims);
        assert(::std::get<2>(s2.dims) == c);
        const auto d = ::std::get<3>(s2.dims);
        return shape(n, h - r + 1, w - s + 1, d);
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;

        using t_p = typename ctx_types::template temp_ctx_type<R>;

        struct temp_ctx_t : t_p {
            const tensor<R, 6> x_;
            const tensor<R, 6> gx_;
            temp_ctx_t(const eval_ctx_t &ctx)
                : t_p(ctx),
                  x_(cosub(ctx.output.shape) * cosub(in<1>(ctx).shape)),
                  gx_(x_.shape)
            {
            }
        };

        static void lowering(const tensor_ref<R, 4> &x,
                             const tensor_ref<R, 6> &x_)
        {
            SET_COUNTER(op_conv_nhwc_lowering);
            R *px_ = x_.data;
            for (auto l : range(dim<0>(x_))) {
                for (auto u : range(dim<1>(x_))) {
                    for (auto v : range(dim<2>(x_))) {
                        for (auto p : range(dim<3>(x_))) {
                            for (auto q : range(dim<4>(x_))) {
                                for (auto k : range(dim<5>(x_))) {
                                    // x_.at(l, u, v, p, q, k) =
                                    *px_++ = x.at(l, p + u, q + v, k);
                                }
                            }
                        }
                    }
                }
            }
        }

        static void uppering(const tensor_ref<R, 4> &x,
                             const tensor_ref<R, 6> &x_)
        {
            SET_COUNTER(op_conv_nhwc_uppering);
            annihilate(x);
            R *px_ = x_.data;
            for (auto l : range(dim<0>(x_))) {
                for (auto u : range(dim<1>(x_))) {
                    for (auto v : range(dim<2>(x_))) {
                        for (auto p : range(dim<3>(x_))) {
                            for (auto q : range(dim<4>(x_))) {
                                for (auto k : range(dim<5>(x_))) {
                                    x.at(l, p + u, q + v, k) += *px_++;
                                    // x_.at(l, u, v, p, q, k);
                                }
                            }
                        }
                    }
                }
            }
        }

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);
            const auto z = ctx.output;
            const auto x_ = ref(t_ctx.x_);
            lowering(x, x_);
            mat_x_mat(
                ref_as(x_, shape(dim(cosub(z.shape)), dim(cosub(y.shape)))), //
                ref_as(y, shape(dim(cosub(y.shape)), dim<3>(y))),            //
                ref_as(z, shape(dim(cosub(z.shape)), dim<3>(z))));
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            const auto y = in<1>(ctx);
            const auto gx = gin<0>(ctx);
            const auto gz = ctx.g_output;
            const auto gx_ = ref(t_ctx.gx_);
            mat_x_mat_trans(
                ref_as(gz, shape(dim(cosub(gz.shape)), dim<3>(gz))),
                ref_as(y, shape(dim(cosub(y.shape)), dim<3>(y))),
                ref_as(gx_, shape(dim(cosub(gz.shape)), dim(cosub(y.shape)))));
            uppering(gx, gx_);
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &t_ctx)
        {
            const auto x_ = ref(t_ctx.x_);
            const auto gy = gin<1>(ctx);
            const auto gz = ctx.g_output;
            mat_trans_x_mat(
                ref_as(x_, shape(dim(cosub(gz.shape)), dim(cosub(gy.shape)))),
                ref_as(gz, shape(dim(cosub(gz.shape)), dim<3>(gz))),
                ref_as(gy, shape(dim(cosub(gy.shape)), dim<3>(gy))));
        }
    };
};
}
