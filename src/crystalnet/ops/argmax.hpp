#pragma once
#include <algorithm>

#include <crystalnet/core/tensor.hpp>

template <typename T> uint32_t argmax(const r_tensor_ref_t<T> &v)
{
    // TODO: check(v.rank() == 1);
    return std::max_element(v.data, v.data + v.shape.dim()) - v.data;
}
