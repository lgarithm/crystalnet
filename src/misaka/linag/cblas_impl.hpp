
// http://www.netlib.org/blas/
#pragma once
#include <cblas.h>
#include <misaka/linag/base.hpp>

template <typename R> struct cblas;

template <> struct cblas<float> {
    static constexpr auto axpy = cblas_saxpy;
    static constexpr auto gemm = cblas_sgemm;
    static constexpr auto gemv = cblas_sgemv;
};

template <> struct cblas<double> {
    static constexpr auto axpy = cblas_daxpy;
    static constexpr auto gemm = cblas_dgemm;
    static constexpr auto gemv = cblas_dgemv;
};

template <typename T> struct cblas_impl {
    static constexpr T alpha = 1;
    static constexpr T beta = 0;
    static constexpr int inc = 1;

    using m_ref_t = matrix_ref_t<T>;
    using v_ref_t = vector_ref_t<T>;
    using blas = cblas<T>;

    static void _gemm(const m_ref_t &a, bool trans_a, //
                      const m_ref_t &b, bool trans_b, const m_ref_t &c)
    {
        blas::gemm(CblasRowMajor,
                   trans_a ? CblasTrans : CblasNoTrans, //
                   trans_b ? CblasTrans : CblasNoTrans, //
                   c.m, c.n, trans_b ? b.m : b.n,       //
                   alpha, a.data, a.n, b.data, b.n, beta, c.data, c.n);
    }

    // a \times b -> C
    static void mm(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        _gemm(a, false, b, false, c);
    }

    // a \times b^T -> C
    static void mmt(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        _gemm(a, false, b, true, c);
    }

    // a^T \times b -> C
    static void mtm(const m_ref_t &a, const m_ref_t &b, const m_ref_t &c)
    {
        _gemm(a, true, b, false, c);
    }

    static void mv(const m_ref_t &a, const v_ref_t &b, const v_ref_t &c)
    {
        blas::gemv(CblasRowMajor, CblasNoTrans, a.m, a.n, alpha, a.data, a.n,
                   b.data, inc, beta, c.data, inc);
    }

    static void vm(const v_ref_t &a, const m_ref_t &b, const v_ref_t &c)
    {
        blas::gemv(CblasRowMajor, CblasTrans, b.m, b.n, alpha, b.data, b.n,
                   a.data, inc, beta, c.data, inc);
    }

    // A + B -> C
    static void vv(const v_ref_t &a, const v_ref_t &b, const v_ref_t &c)
    {
        // blas::axpy();
    }
};
