#include <misaka/core/gc.hpp>
#include <misaka/model/operator.hpp>

static auto gc = GC<operator_t>();

operator_t *register_op(const char *const name, uint8_t arity,
                        shape_func_t infer, forward_func_t eval,
                        backward_func_t feed)
{
    return gc(new operator_t(name, arity, infer, eval, feed));
}
