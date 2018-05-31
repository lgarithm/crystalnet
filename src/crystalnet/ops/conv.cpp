#include <crystalnet.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/ops/conv_nhwc_generic.hpp>

const operator_t *make_op_conv2d(uint32_t padding_h, uint32_t padding_w,
                                 uint32_t stride_h, uint32_t stride_w)
{
    const auto op = gc(new op_conv2d_impl_t(
        conv_nhwc_generic::trait_t(r_shape(padding_h, padding_w),  //
                                   r_shape(stride_h, stride_w))));
    return _register_generic_bi_op(op);
}
