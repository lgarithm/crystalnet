#include <crystalnet.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/ops/pool_generic.hpp>

const operator_t *make_op_pool2d(uint32_t r, uint32_t s,  //
                                 uint32_t stride_r, uint32_t stride_s)
{
    const auto op = gc(
        new op_pool2d_impl_t(pool2d_c::trait_t(r_shape(r, s),  //
                                               r_shape(stride_r, stride_s))));
    return _register_generic_bi_op(op);
}
