#pragma once
#include <cassert>

#include <crystalnet/linag/base.hpp>

template <typename T> struct plain_impl {
    using m_ref_t = matrix_ref_t<T>;
    using v_ref_t = vector_ref_t<T>;

    // a \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.m
    static void mm(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        const auto m = equally(a.m, c.m);
        const auto n = equally(b.n, c.n);
        const auto l = equally(a.n, b.m);
        for (auto i = 0; i < m; ++i) {
            for (auto j = 0; j < n; ++j) {
                T tmp = 0;
                for (auto k = 0; k < l; ++k) {
                    tmp += a(i, k) * b(k, j);
                }
                c(i, j) = tmp;
            }
        }
    }

    // a \times b^T -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.n
    static void mmt(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        const auto m = equally(a.m, c.m);
        const auto n = equally(b.m, c.n);
        const auto l = equally(a.n, b.n);
        for (auto i = 0; i < m; ++i) {
            for (auto j = 0; j < n; ++j) {
                T tmp = 0;
                for (auto k = 0; k < l; ++k) {
                    tmp += a(i, k) * b(j, k);
                }
                c(i, j) = tmp;
            }
        }
    }

    // a^T \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.m == b.m
    static void mtm(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        const auto m = equally(a.n, c.m);
        const auto n = equally(b.n, c.n);
        const auto l = equally(a.m, b.m);
        for (auto i = 0; i < m; ++i) {
            for (auto j = 0; j < n; ++j) {
                T tmp = 0;
                for (auto k = 0; k < l; ++k) {
                    tmp += a(k, i) * b(k, j);
                }
                c(i, j) = tmp;
            }
        }
    }

    // a \times b -> c where a[m, n], b[n] -> c[n]; a.n = b.n
    static void mv(const m_ref_t &a, const v_ref_t &b, const v_ref_t &c)
    {
        const auto m = equally(a.m, c.n);
        const auto n = equally(a.n, b.n);
        for (auto i = 0; i < m; ++i) {
            T tmp = 0;
            for (auto j = 0; j < n; ++j) {
                tmp += a(i, j) * b(j);
            }
            c(i) = tmp;
        }
    }

    // [1, n] X [n, m] -> [1, m]
    // a \times b -> c where a[n], b[m, n] -> c[n]; a.n = b.m
    static void vm(const v_ref_t &a, const m_ref_t &b, const v_ref_t &c)
    {
        const auto m = equally(a.n, b.m);
        const auto n = equally(b.n, c.n);
        for (auto i = 0; i < n; ++i) {
            T tmp = 0;
            for (auto j = 0; j < m; ++j) {
                tmp += a(j) * b(j, i);
            }
            c(i) = tmp;
        }
    }

    // a + b -> c
    static void vv(const v_ref_t &a, const v_ref_t &b, const v_ref_t &c)
    {
        const auto l = equally(a.n, b.n, c.n);
        for (auto i = 0; i < l; ++i) {
            c.data[i] = a.data[i] + b.data[i];
        }
    }
};
