#pragma once

namespace tea
{
template <typename T> struct range_t {
    const T from;
    const T to;

    explicit range_t(T n) : from(0), to(n) {}

    explicit range_t(T m, T n) : from(m), to(n) {}

    struct iterator {
        T pos;

        explicit iterator(T pos) : pos(pos) {}

        bool operator!=(const iterator &it) const { return pos != it.pos; }

        T operator*() const { return pos; }

        void operator++() { ++pos; }
    };

    auto begin() const { return iterator(from); }

    auto end() const { return iterator(to); }
};

template <typename T> range_t<T> range(T n) { return range_t<T>(n); }

template <typename T> range_t<T> range(T m, T n) { return range_t<T>(m, n); }
}
