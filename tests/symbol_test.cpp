#include <cassert>

#include <crystalnet.h>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/symbol/model.hpp>
#include <crystalnet/symbol/node.hpp>

void test_1()
{
    s_model_ctx_t ctx;
    auto x = ctx.make_placeholder(shape_t(10));
    auto p = ctx.make_parameter(shape_t(10, 100));
    auto y = ctx.make_operator(*op_mul, x, p);
}

int main()
{
    test_1();
    return 0;
}
