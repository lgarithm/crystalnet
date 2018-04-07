#include <crystalnet.h>
#include <crystalnet/ops/add.hpp>
#include <crystalnet/ops/conv_nhwc.hpp>
#include <crystalnet/ops/mul.hpp>
#include <crystalnet/ops/pool.hpp>
#include <crystalnet/ops/relu.hpp>
#include <crystalnet/ops/softmax.hpp>
#include <crystalnet/ops/xentropy.hpp>

const operator_t *op_add = _register_bi_op<add>("add");
const operator_t *op_mul = _register_bi_op<mul>("mul");
const operator_t *op_pool2d_c_max =
    _register_bi_op<pool2d_n_c_max>("pool2d_n_c_max");
const operator_t *op_relu = _register_bi_op<relu>("relu");
const operator_t *op_softmax = _register_bi_op<softmax>("softmax");
const operator_t *op_xentropy = _register_bi_op<xentropy>("cross entropy");
const operator_t *op_conv_nhwc = _register_bi_op<conv2d>("conv2d");
