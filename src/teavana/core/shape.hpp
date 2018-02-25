#pragma once

#include <array>       // for array
#include <cassert>     // for assert
#include <cstddef>     // for size_t
#include <cstdint>     // for uint8_t
#include <cstdio>      // for sprintf, perror
#include <cstring>     // for strlen
#include <functional>  // for minus, plus
#include <string>      // for string
#include <type_traits> // for enable_if
#include <utility>     // for make_index_sequence, index_sequence

namespace tea
{
template <uint8_t r> struct shape_t {
    static constexpr uint8_t rank = r;
    const ::std::array<size_t, r> dims;

    constexpr explicit shape_t(const ::std::array<size_t, r> &dims) : dims(dims)
    {
    }

    template <typename... Args,
              typename = typename ::std::enable_if<sizeof...(Args) == r>::type>
    constexpr explicit shape_t(Args... args)
        : dims{static_cast<size_t>(args)...}
    {
    }

    template <typename... Args,
              typename = typename ::std::enable_if<sizeof...(Args) == r>::type>
    size_t idx(Args... args) const
    {
        const ::std::array<size_t, r> offs{static_cast<size_t>(args)...};
        size_t off = 0;
        for (uint8_t i = 0; i < r; ++i) {
            off = off * dims[i] + offs[i];
        }
        return off;
    }
};

template <typename... Args>
constexpr shape_t<sizeof...(Args)> shape(Args... args)
{
    return shape_t<sizeof...(Args)>{static_cast<size_t>(args)...};
}

template <typename T1, typename T2, size_t r, size_t... Is>
constexpr auto index_array(const T2 *arr, ::std::index_sequence<Is...>)
{
    return ::std::array<T1, r>{arr[Is]...};
}

template <uint8_t r, typename T> shape_t<r> shape(const T *dims)
{
    return shape_t<r>(
        index_array<size_t, T, r>(dims, ::std::make_index_sequence<r>()));
}

template <uint8_t r> constexpr shape_t<r> unit_shape();

template <> constexpr shape_t<0> unit_shape() { return shape(); }

template <uint8_t r> constexpr shape_t<r> unit_shape()
{
    return shape(1) * unit_shape<r - 1>();
}

constexpr shape_t<2> transpose(const shape_t<2> &s)
{
    return shape(::std::get<1>(s.dims), ::std::get<0>(s.dims));
}

template <uint8_t r>
constexpr bool operator==(const shape_t<r> &s, const shape_t<r> &t)
{
    // return s.dims == t.dims;
    for (auto i = 0; i < r; ++i) {
        if (s.dims[i] != t.dims[i]) {
            return false;
        }
    }
    return true;
}

template <uint8_t r>
constexpr bool operator<=(const shape_t<r> &s, const shape_t<r> &t)
{
    for (auto i = 0; i < r; ++i) {
        if (s.dims[i] > t.dims[i]) {
            return false;
        }
    }
    return true;
}

template <typename T, typename F, size_t r, size_t... Is>
constexpr auto zip_with(const ::std::array<T, r> &a,
                        const ::std::array<T, r> &b, const F &f,
                        ::std::index_sequence<Is...>)
{
    return ::std::array<T, r>{f(::std::get<Is>(a), ::std::get<Is>(b))...};
}

template <uint8_t r>
constexpr shape_t<r> operator+(const shape_t<r> &s, const shape_t<r> &t)
{
    return shape_t<r>(zip_with(s.dims, t.dims, ::std::plus<size_t>(),
                               ::std::make_index_sequence<r>()));
}

template <uint8_t r>
constexpr shape_t<r> operator-(const shape_t<r> &s, const shape_t<r> &t)
{
    return shape_t<r>(zip_with(s.dims, t.dims, ::std::minus<size_t>(),
                               ::std::make_index_sequence<r>()));
}

template <uint8_t r> constexpr size_t len(const shape_t<r> &s)
{
    return ::std::get<0>(s.dims);
}

template <uint8_t r> constexpr size_t wid(const shape_t<r> &s)
{
    return ::std::get<1>(s.dims);
}

template <uint8_t r> constexpr size_t dim(const shape_t<r> &s)
{
    size_t d = 1;
    for (auto i : s.dims) {
        d *= i;
    }
    return d;
};

template <size_t off, typename T, size_t r, size_t... Is>
constexpr auto shift_idx(const ::std::array<T, r> &arr,
                         ::std::index_sequence<Is...>)
{
    return ::std::array<T, r - 1>{::std::get<Is + off>(arr)...};
}

template <uint8_t r> constexpr shape_t<r - 1> sub(const shape_t<r> &s)
{
    return shape_t<r - 1>(
        shift_idx<1>(s.dims, ::std::make_index_sequence<r - 1>()));
}

template <uint8_t r> constexpr shape_t<r - 1> cosub(const shape_t<r> &s)
{
    return shape_t<r - 1>(
        shift_idx<0>(s.dims, ::std::make_index_sequence<r - 1>()));
}

template <typename T, size_t p, size_t q, size_t... Is, size_t... Js>
constexpr auto
merge_idx(const ::std::array<T, p> &a, ::std::index_sequence<Is...>,
          const ::std::array<T, q> &b, ::std::index_sequence<Js...>)
{
    return ::std::array<T, p + q>{::std::get<Is>(a)..., ::std::get<Js>(b)...};
}

template <uint8_t p, uint8_t q>
constexpr shape_t<p + q> operator*(const shape_t<p> &s, const shape_t<q> &t)
{
    return shape_t<p + q>(merge_idx(s.dims, ::std::make_index_sequence<p>(),
                                    t.dims, ::std::make_index_sequence<q>()));
}

template <uint8_t r>::std::string to_str(const shape_t<r> &s)
{
    static char buffer[r * 22 + 3];
    char *p = buffer;
    sprintf(p, "(");
    p += strlen(p);
    for (uint8_t i = 0; i < r; ++i) {
        if (i > 0) {
            sprintf(p, ", ");
            p += strlen(p);
        }
        sprintf(p, "%lu", s.dims[i]);
        p += strlen(p);
    }
    sprintf(p, ")");
    return buffer;
}

template <uint8_t p, uint8_t q>
size_t assert_eq_dim(const shape_t<p> &s, const shape_t<q> &t, const char *name)
{
    const size_t n = dim(s);
    if (n != dim(t)) {
        auto msg = "dim" + to_str(s) + " != dim" + to_str(t) + " in " + name;
        perror(msg.c_str());
        assert(false);
    }
    return n;
}
}
