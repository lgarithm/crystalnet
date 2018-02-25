#include <cassert>
#include <misaka.h>
#include <misaka/core/tensor.hpp>
#include <vector>

tensor_t *new_tensor(shape_t *shape, uint8_t dtype)
{
    return new tensor_t(*shape, dtype);
}

void free_tensor(tensor_t *tensor) { delete tensor; }

const shape_t *tensor_shape(tensor_t *tensor) { return &tensor->shape; }

tensor_ref_t tensor_t::operator[](uint32_t idx) const
{
    assert(idx < this->shape.len());
    if (shape.rank() == 0) {
        return ref(*this);
    }

    shape_t shape(std::vector<uint32_t>(this->shape.dims.begin() + 1,
                                        this->shape.dims.end()));
    uint32_t offset = idx * shape.dim() * dtype_size(this->dtype);
    return tensor_ref_t(shape, this->dtype, (uint8_t *)(this->data) + offset);
}
