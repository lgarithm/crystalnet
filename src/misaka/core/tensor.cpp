#include <misaka.h>
#include <misaka/core/tensor.hpp>

tensor_t *new_tensor(shape_t *shape, uint8_t dtype)
{
    return new tensor_t(*shape, dtype);
}

void free_tensor(tensor_t *tensor) { delete tensor; }

const shape_t *tensor_shape(tensor_t *tensor) { return &tensor->shape; }
