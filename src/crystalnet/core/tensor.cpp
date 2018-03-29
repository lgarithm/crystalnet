#include <crystalnet.h>
#include <crystalnet/core/tensor.hpp>

tensor_t::tensor_t(const shape_t &shape, uint8_t dtype)
    : _tensor_meta_t(dtype, shape),
      _data(new uint8_t[dtype_size(dtype) * shape.dim()]), data(_data.get()),
      self(ref(*this))
{
    // LOG_TENSOR_USAGE(shape, dtype_size(dtype));
    memset(data, 0, dtype_size(dtype) * shape.dim());
}

tensor_t *new_tensor(const shape_t *shape, uint8_t dtype)
{
    return new tensor_t(*shape, dtype);
}

void del_tensor(const tensor_t *tensor) { delete tensor; }

tensor_ref_t ref(const tensor_t &tensor)
{
    return tensor_ref_t(tensor.shape, tensor.dtype, tensor.data);
}

const uint8_t tensor_dtype(const tensor_ref_t *tensor) { return tensor->dtype; }

const shape_t *tensor_shape(const tensor_ref_t *tensor)
{
    return &tensor->shape;
}

const tensor_ref_t *tensor_ref(const tensor_t *tensor) { return &tensor->self; }

void *tensor_data_ptr(const tensor_ref_t *r) { return r->data; }
