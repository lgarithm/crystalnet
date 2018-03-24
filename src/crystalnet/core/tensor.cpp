#include <crystalnet.h>
#include <crystalnet/core/tensor.hpp>

tensor_t *new_tensor(const shape_t *shape, uint8_t dtype)
{
    return new tensor_t(*shape, dtype);
}

void del_tensor(const tensor_t *tensor) { delete tensor; }

const tensor_ref_t *new_tensor_ref(const tensor_t *tensor)
{
    return new tensor_ref_t(*tensor);
}

void del_tensor_ref(const tensor_ref_t *r) { delete r; }

void *tensor_data_ptr(const tensor_ref_t *r) { return r->data; }

const shape_t *tensor_shape(tensor_t *tensor) { return &tensor->shape; }
