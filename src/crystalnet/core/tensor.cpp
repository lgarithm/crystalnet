#include <crystalnet.h>
#include <crystalnet/core/tensor.hpp>

tensor_t *new_tensor(const shape_t *shape, uint8_t dtype)
{
    return new tensor_t(*shape, dtype);
}

void del_tensor(tensor_t *tensor) { delete tensor; }

const shape_t *tensor_shape(tensor_t *tensor) { return &tensor->shape; }
