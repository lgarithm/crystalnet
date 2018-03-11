#include <crystalnet/ops/add.hpp>
#include <crystalnet/ops/conv_nhwc.hpp>
#include <crystalnet/ops/mul.hpp>
#include <crystalnet/ops/pool.hpp>
#include <crystalnet/ops/relu.hpp>
#include <crystalnet/ops/softmax.hpp>
#include <crystalnet/ops/xentropy.hpp>

operator_t *op_pool2d_c_max = _register_bi_op<pool2d_n_c_max>("pool2d_n_c_max");
operator_t *op_relu = _register_bi_op<relu>("relu");
operator_t *op_softmax = _register_bi_op<softmax>("softmax");
