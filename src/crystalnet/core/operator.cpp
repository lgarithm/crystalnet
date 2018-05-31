#include <crystalnet-internal.h>
#include <crystalnet/core/operator.hpp>

operator_registry_t operator_registry;

const operator_t *register_op(const char *const name, uint8_t arity,
                              shape_func_t *infer, forward_func_t *eval,
                              backward_func_t *feed)
{
    return operator_registry.own(new operator_t(name, arity, infer, eval, feed),
                                 name);
}

shape_t *infer(const operator_t *op, const shape_list_t *shape_list)
{
    shape_t shape = (*op->infer)(*shape_list);
    return new shape_t(shape);
}
