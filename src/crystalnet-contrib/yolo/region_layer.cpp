#include <cmath>

#include <crystalnet-contrib/darknet/darknet.h>
#include <crystalnet-contrib/yolo/activation.hpp>
#include <crystalnet-contrib/yolo/cmath.h>
#include <crystalnet-contrib/yolo/region_layer.h>
#include <crystalnet-contrib/yolo/yolo.hpp>
#include <crystalnet-internal.h>
#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/layer.hpp>
#include <crystalnet/core/operator.hpp>  // TODO: don't include private headers
#include <crystalnet/utility/range.hpp>

// TODO: use proxy tensor
template <typename T> struct proxy_array_t {
    T *const _data;
    const uint32_t _n;
    const uint32_t stride;

    proxy_array_t(T *data, uint32_t n, uint32_t stride)
        : _data(data), _n(n), stride(stride)
    {
    }

    T &operator[](int i) const { return _data[i * stride]; }
};

namespace darknet::yolov2
{
template <typename T>
void softmax(uint32_t n, uint32_t stride, T *input, T *output)
{
    proxy_array_t<T> x(input, n, stride);
    proxy_array_t<T> y(output, n, stride);

    T x_max = std::numeric_limits<T>::lowest();
    for (auto i : range(n)) { x_max = std::max(x_max, x[i]); }

    T sum = 0;
    for (auto i : range(n)) {
        const T e = c_exp(x[i] - x_max);
        sum += e;
        y[i] = e;
    }

    for (auto i : range(n)) { y[i] /= sum; }
}

struct region_op_t {
    constexpr static uint8_t arity = 2;

    const int n;        // 5
    const int classes;  // 80
    const int coords;   // 4
    const int m;        // m == coords + 1 + classes;

    const int h;  // 13
    const int w;  // 13

    using T = float;

    region_op_t(int h, int w, int n, int classes, int coords)
        : h(h), w(w),                                                //
          n(n),                                                      //
          classes(classes), coords(coords), m(coords + 1 + classes)  //
    {
    }

    shape_t infer(const shape_list_t &shape_list) const
    {
        const auto [p, q] = cast<arity>(shape_list.shapes, auto_hint);
        if (p.rank() == 3) {
            return _infer(shape_list_t({p.batch(1), q}));
        } else {
            return _infer(shape_list);
        }
    }

    shape_t _infer(const shape_list_t &shape_list) const
    {
        const auto [p, q] = cast<arity>(shape_list.shapes, auto_hint);
        const auto [b, c, height, width] = cast<4>(p.dims, auto_hint);
        check(this->h == height);
        check(this->w == width);
        check(c == n * m);
        check(q == shape_t(n, 2));
        return p;
    }

    void forward(const forward_ctx_t &ctx) const
    {
        const auto [p, q] = cast<arity>(ctx.inputs.shapes().shapes, auto_hint);
        const auto [batch_size, c, h, w] = cast<4>(p.dims, auto_hint);
        check(c == n * m);
        // batch_size == 1
        const shape_t new_shape(batch_size, n, m, h, w);

        const auto [_x, _bias] = cast<arity>(ctx.inputs._args, auto_hint);
        const auto _y = ctx.output;

        _y.copy_from(_x);
        const auto x = ranked<5, T>(_x.reshape(new_shape));
        const auto y = ranked<5, T>(_y.reshape(new_shape));

        // const auto act = logistic<T>();
        for (auto b : range(batch_size)) {
            for (auto nn : range(n)) {
                const auto yy = y[b][nn];  // [C + 1 + K, h, w]
                const uint32_t stride = h * w;
                for (auto i : range(h)) {
                    for (auto j : range(w)) {
                        yy.at(0, i, j) = c_logistic(yy.at(0, i, j));
                        yy.at(1, i, j) = c_logistic(yy.at(1, i, j));
                        yy.at(4, i, j) = c_logistic(yy.at(4, i, j));
                        const auto data = yy[5].data + i * w + j;
                        softmax(classes, stride, data, data);
                    }
                }
            }
        }
    }

    void backward(const backward_ctx_t &ctx) const
    {
        throw std::logic_error("NOT IMPLEMENTED");
    }
};

struct region_layer : s_layer_t {
    const int n;        // 5
    const int classes;  // 80
    const int coords;   // 4

    const int h;  // 13
    const int w;  // 13

    const operator_t *op;

    region_layer(int h, int w, int n, int classes, int coords)
        : h(h), w(w), n(n), classes(classes), coords(coords),
          op(_register_generic_bi_op(
              gc(new region_op_t(h, w, n, classes, coords)), "darknet::region"))
    {
    }

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        const std::string prefix = name_prefix(ctx);
        symbol anchors = ctx.make_parameter(shape_t(n, 2), prefix + "_anchors");
        return ctx.make_operator(*op, {x, anchors},
                                 "region_" +
                                     std::to_string(get_layer_number(ctx)));
    }
};
}  // namespace darknet::yolov2

s_layer_t *make_region_layer(int w, int h, int n, int classes, int coords)
{
    return new darknet::yolov2::region_layer(h, w, n, classes, coords);
}
