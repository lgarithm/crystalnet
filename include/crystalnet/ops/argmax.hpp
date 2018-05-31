#pragma once
#include <algorithm>
#include <vector>

#include <crystalnet/core/tensor.hpp>
#include <crystalnet/utility/range.hpp>

template <typename T> uint32_t argmax(const r_tensor_ref_t<T> &v)
{
    // TODO: check(v.rank() == 1);
    return std::max_element(v.data, v.data + v.shape.dim()) - v.data;
}

template <typename T>
void top_indexes(const vector_ref_t<T> &x, const vector_ref_t<int32_t> &y)
{
    const uint32_t n = x.shape.dim();
    const uint32_t k = y.shape.dim();
    check(k <= n);
    std::vector<std::pair<T, int32_t>> z(n);
    for (auto i : range(n)) {
        z[i] = std::make_pair(x.data[i], i);
    }
    std::sort(z.begin(), z.end());
    for (auto i : range(k)) {
        y.data[i] = z[n - 1 - i].second;
    }
}
