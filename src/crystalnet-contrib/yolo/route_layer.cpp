#include <crystalnet-contrib/yolo/route_layer.h>
#include <crystalnet-internal.h>
#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/operator.hpp>  // TODO: don't include private headers
#include <crystalnet/core/user_context.hpp>
#include <crystalnet/symbol/model.hpp>
#include <crystalnet/utility/range.hpp>

namespace darknet
{
struct route_1 {
    constexpr static uint8_t arity = 1;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto [p] = cast<arity>(shape_list.shapes, auto_hint);
        return p;
    }

    using T = float;  // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            const auto [x] = cast<arity>(inputs._args, auto_hint);
            output.copy_from(x);
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            // TODO
        }
    };
};

struct route_2 {
    constexpr static uint8_t arity = 2;

    static shape_t infer(const shape_list_t &shape_list)
    {
        const auto [p, q] = cast<arity>(shape_list.shapes, auto_hint);
        if (p.rank() == 3) {
            return _infer(shape_list_t({p.batch(1), q.batch(1)}));
        } else {
            return _infer(shape_list);
        }
    }

    static shape_t _infer(const shape_list_t &shape_list)
    {
        const auto [p, q] = cast<arity>(shape_list.shapes, auto_hint);
        const auto [b, c1, h, w] = cast<4>(p.dims, auto_hint);
        const auto [_b, c2, _h, _w] = cast<4>(q.dims, auto_hint);
        check(h == _h);
        check(w == _w);
        check(b == _b);
        return shape_t(b, c1 + c2, h, w);
    }

    using T = float;  // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        // [N, C, H, W], [N, D, H, W] -> [N, C + D, H, W]
        void operator()() const
        {
            const auto [x, y] = cast<arity>(inputs._args, auto_hint);
            const auto [n, c, h, w] = cast<4>(x.shape.dims, auto_hint);
            const auto [_n, d, _h, _w] = cast<4>(y.shape.dims, auto_hint);
            for (const auto b : range(n)) {
                const auto xx = x[b];
                const auto yy = y[b];
                const auto zz = output[b];
                // xx :: [C, H, W], yy :: [D, H, W] -> zz :: [C + D, H, W]
                const auto zz_1 = zz.slice(0, c);
                const auto zz_2 = zz.slice(c, c + d);
                zz_1.copy_from(xx);
                zz_2.copy_from(yy);
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const { throw std::logic_error("NOT IMPLEMENTED"); }
    };
};

}  // namespace darknet

const operator_t *route_1_opl =
    _register_bi_op<darknet::route_1>("darknet::route_1");
const operator_t *route_2_opl =
    _register_bi_op<darknet::route_2>("darknet::route_2");

symbol route_1(context_t *ctx, symbol p)
{
    return ctx->_layers(ctx->make_operator(*route_1_opl, {p}, "route_1"));
}

symbol route_2(context_t *ctx, symbol p1, symbol p2)
{
    symbol l = ctx->make_operator(*route_2_opl, {p1, p2}, "route_2");
    return ctx->_layers(l);
}
