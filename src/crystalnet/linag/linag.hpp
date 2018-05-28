#pragma once
#include <crystalnet/linag/plain_impl.hpp>

#ifdef CRYSTALNET_USE_CBLAS
#include <crystalnet/linag/cblas_impl.hpp>
template <typename T> using default_engine = cblas_impl<T>;
#else
template <typename T> using default_engine = plain_impl<T>;
#endif

template <typename T, typename engine = default_engine<T>> struct linag {
    using m_ref_t = matrix_ref_t<T>;
    using v_ref_t = vector_ref_t<T>;

    // a \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.m
    static void mm(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        engine::mm(a, b, c);
    }

    // a \times b^T -> c where a[m, n], b[m, n] -> c[m, n]; a.n == b.n
    static void mmt(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        engine::mmt(a, b, c);
    }

    // a^T \times b -> c where a[m, n], b[m, n] -> c[m, n]; a.m == b.m
    static void mtm(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        engine::mtm(a, b, c);
    }

    // a \times b -> c where a[m, n], b[n] -> c[n]; a.n = b.n
    static void mv(const m_ref_t &a, const v_ref_t &b, const v_ref_t &c)
    {
        engine::mv(a, b, c);
    }

    // [1, n] X [n, m] -> [1, m]
    // a \times b -> c where a[n], b[m, n] -> c[n]; a.n = b.m
    static void vm(const v_ref_t &a, const m_ref_t &b, const v_ref_t &c)
    {
        engine::vm(a, b, c);
    }

    // a + b -> c
    static void vv(const v_ref_t &a, const v_ref_t &b, const v_ref_t &c)
    {
        engine::vv(a, b, c);
    }
};
