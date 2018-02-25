#pragma once

#include <cinttypes> // for PRId64
#include <cstdint>   // for uint8_t, uint64_t
#include <cstdio>    // for printf, sprintf
#include <ratio>     // for ratio
#include <string>    // for string

namespace std
{
using kibi = ::std::ratio<1LL << 10, 1LL>;
using mebi = ::std::ratio<1LL << 20, 1LL>;
using gibi = ::std::ratio<1LL << 30, 1LL>;
}

namespace tea
{
template <typename R, uint8_t r> struct tensor;
template <typename R, uint8_t r> struct tensor_ref;

template <typename Show> const char *p_str(const Show &s)
{
    return to_str(s).c_str();
}

template <typename Show> void print(const Show &s) { printf("%s\n", p_str(s)); }

inline void print(const uint8_t &x) { printf("%d", x); }

inline void print(const float &x) { printf("%f", x); }

template <typename T, uint8_t r> struct printer;

template <typename T> struct printer<T, 0> {
    void operator()(const tensor_ref<T, 0> &t) { print(scalar(t)); }
};

template <typename T> struct printer<T, 1> {
    void operator()(const tensor_ref<T, 1> &t)
    {
        printf("[");
        bool fst = true;
        for (const auto &it : t) {
            if (!fst) {
                printf(",");
            }
            fst = false;
            print(it);
        }
        printf("]");
    }
};

template <typename T, uint8_t r> struct printer {
    void operator()(const tensor_ref<T, r> &t)
    {
        printer<T, r - 1> p1;
        printf("[");
        bool fst = true;
        for (const auto &it : t) {
            if (!fst) {
                printf(",");
            }
            fst = false;
            p1(it);
        }
        printf("]");
    }
};

template <typename T, uint8_t r> void print(const tensor_ref<T, r> &t)
{
    printer<T, r>()(t);
    printf("\n");
}

template <typename T, uint8_t r> void print(const tensor<T, r> &t)
{
    printer<T, r>()(ref(t));
    printf("\n");
}

template <typename T, typename V> T constexpr value()
{
    return (T)V::num / V::den;
}

template <typename T> static constexpr T kibi = value<T, ::std::kibi>();
template <typename T> static constexpr T mebi = value<T, ::std::mebi>();
template <typename T> static constexpr T gibi = value<T, ::std::gibi>();

inline ::std::string size2str(uint64_t n)
{
    char buffer[64];
    if (n > gibi<uint64_t>) {
        sprintf(buffer, "%.2fG", n / gibi<float>);
    } else if (n > mebi<uint64_t>) {
        sprintf(buffer, "%.2fM", n / mebi<float>);
    } else if (n > kibi<uint64_t>) {
        sprintf(buffer, "%.2fK", n / kibi<float>);
    } else {
        sprintf(buffer, "%" PRId64, n);
    }
    return buffer;
}
}
