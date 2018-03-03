#include <cstdio>

#include <crystalnet/core/gc.hpp>
#include <crystalnet/model/operator.hpp>

static auto gc = GC<operator_t>();

operator_t *register_op(const char *const name, uint8_t arity,
                        shape_func_t infer, forward_func_t eval,
                        backward_func_t feed)
{
    printf("[D] registering operator: %s\n", name);
    return gc(new operator_t(name, arity, infer, eval, feed));
}
