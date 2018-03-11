#pragma once
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

template <typename T, size_t... i>
auto _index(const std::vector<T> &v, std::index_sequence<i...>)
{
    return std::array<T, sizeof...(i)>({v[i]...});
}

template <uint8_t rank, typename T> auto cast(const std::vector<T> &v)
{
    // TODO: log a runtime error
    check(v.size() == rank);
    return _index(v, std::make_index_sequence<rank>());
}
