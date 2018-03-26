#pragma once
#include <crystalnet/core/error.hpp>
#include <crystalnet/core/tensor.hpp>

template <typename T> T equally(T m, T n)
{
    check(m == n);
    return m;
}

template <typename T> T equally(T k, T m, T n)
{
    check(k == m);
    check(m == n);
    return k;
}

template <typename T> struct plain_impl {
    using m_ref_t = matrix_ref_t<T>;
    using v_ref_t = vector_ref_t<T>;

    // a \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.m
    static void mm(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        const auto m = equally(len(a), len(c));
        const auto n = equally(wid(b), wid(c));
        const auto l = equally(wid(a), len(b));
        for (auto i = 0; i < m; ++i) {
            for (auto j = 0; j < n; ++j) {
                T tmp = 0;
                for (auto k = 0; k < l; ++k) {
                    tmp += a.at(i, k) * b.at(k, j);
                }
                c.at(i, j) = tmp;
            }
        }
    }

    // a \times b^T -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.n
    static void mmt(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        const auto m = equally(len(a), len(c));
        const auto n = equally(len(b), wid(c));
        const auto l = equally(wid(a), wid(b));
        for (auto i = 0; i < m; ++i) {
            for (auto j = 0; j < n; ++j) {
                T tmp = 0;
                for (auto k = 0; k < l; ++k) {
                    tmp += a.at(i, k) * b.at(j, k);
                }
                c.at(i, j) = tmp;
            }
        }
    }

    // a^T \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.m == b.m
    static void mtm(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        const auto m = equally(wid(a), len(c));
        const auto n = equally(wid(b), wid(c));
        const auto l = equally(len(a), len(b));
        for (auto i = 0; i < m; ++i) {
            for (auto j = 0; j < n; ++j) {
                T tmp = 0;
                for (auto k = 0; k < l; ++k) {
                    tmp += a.at(k, i) * b.at(k, j);
                }
                c.at(i, j) = tmp;
            }
        }
    }

    // a \times b -> c where a[m, n], b[n] -> c[n]; a.n = b.n
    static void mv(const m_ref_t &a, const v_ref_t &b, const v_ref_t &c)
    {
        const auto m = equally(len(a), len(c));
        const auto n = equally(wid(a), len(b));
        for (auto i = 0; i < m; ++i) {
            T tmp = 0;
            for (auto j = 0; j < n; ++j) {
                tmp += a.at(i, j) * b.at(j);
            }
            c.at(i) = tmp;
        }
    }

    // [1, n] X [n, m] -> [1, m]
    // a \times b -> c where a[n], b[m, n] -> c[n]; a.n = b.m
    static void vm(const v_ref_t &a, const m_ref_t &b, const v_ref_t &c)
    {
        const auto m = equally(len(a), len(b));
        const auto n = equally(wid(b), len(c));
        for (auto i = 0; i < n; ++i) {
            T tmp = 0;
            for (auto j = 0; j < m; ++j) {
                tmp += a.at(j) * b.at(j, i);
            }
            c.at(i) = tmp;
        }
    }

    // a + b -> c
    static void vv(const v_ref_t &a, const v_ref_t &b, const v_ref_t &c)
    {
        const auto l = equally(len(a), len(b), len(c));
        for (auto i = 0; i < l; ++i) {
            c.data[i] = a.data[i] + b.data[i];
        }
    }
};
