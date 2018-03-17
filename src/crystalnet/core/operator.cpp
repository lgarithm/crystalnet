#include <cstdio>

#include <crystalnet.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/core/operator.hpp>

static auto gc = GC<operator_t>();

operator_t *register_op(const char *const name, uint8_t arity,
                        shape_func_t *infer, forward_func_t *eval,
                        backward_func_t *feed)
{
    // printf("[D] registering operator: %s\n", name);
    return gc(new operator_t(name, arity, infer, eval, feed));
}

shape_t *infer(const operator_t *op, const shape_list_t *shape_list)
{
    shape_t shape = (*op->infer)(*shape_list);
    return new shape_t(shape);
}
