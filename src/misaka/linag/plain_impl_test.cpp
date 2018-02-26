#include <algorithm>
#include <cassert>
#include <misaka/core/shape.hpp>
#include <misaka/core/tensor.hpp>
#include <misaka/linag/plain_impl.hpp>

template <typename T, typename linag> struct test_linag {
    static constexpr auto dtype = idx_type<T>::type;

    static auto mref(const tensor_t &tensor)
    {
        return as_matrix_ref<T>(ref(tensor));
    }

    static auto vref(const tensor_t &tensor)
    {
        return as_vector_ref<T>(ref(tensor));
    }

    static void test_mm(uint32_t k, uint32_t m, uint32_t n)
    {
        tensor_t a(shape_t(k, m), dtype);
        tensor_t b(shape_t(m, n), dtype);
        tensor_t c(shape_t(k, n), dtype);
        linag::mm(mref(a), mref(b), mref(c));
    }

    static void test_mmt(uint32_t k, uint32_t m, uint32_t n)
    {
        tensor_t a(shape_t(k, m), dtype);
        tensor_t b(shape_t(n, m), dtype);
        tensor_t c(shape_t(k, n), dtype);
        linag::mmt(mref(a), mref(b), mref(c));
    }

    static void test_mtm(uint32_t k, uint32_t m, uint32_t n)
    {
        tensor_t a(shape_t(m, k), dtype);
        tensor_t b(shape_t(m, n), dtype);
        tensor_t c(shape_t(k, n), dtype);
        linag::mtm(mref(a), mref(b), mref(c));
    }

    static void test_mv(uint32_t m, uint32_t n)
    {
        tensor_t a(shape_t(m, n), dtype);
        tensor_t b(shape_t(n), dtype);
        tensor_t c(shape_t(m), dtype);
        linag::mv(mref(a), vref(b), vref(c));
    }

    static void test_vm(uint32_t m, uint32_t n)
    {
        tensor_t a(shape_t(m), dtype);
        tensor_t b(shape_t(m, n), dtype);
        tensor_t c(shape_t(n), dtype);
        linag::vm(vref(a), mref(b), vref(c));
    }

    static void test_vv(uint32_t n)
    {
        tensor_t a(shape_t(n), dtype);
        tensor_t b(shape_t(n), dtype);
        tensor_t c(shape_t(n), dtype);
        linag::vv(vref(a), vref(b), vref(c));
    }

    static void test_k_m_n(void(test_func)(uint32_t, uint32_t, uint32_t))
    {
        uint32_t dimss[][3] = {
            {1, 1, 1},
            {10, 10, 10},
            {1, 100, 1},
            {100, 1, 100},
        };
        for (auto dims : dimss) {
            test_func(dims[0], dims[1], dims[2]);
        }
        uint32_t dims[] = {2, 3, 4};
        do {
            test_func(dims[0], dims[1], dims[2]);
        } while (std::next_permutation(dims, dims + 3));
    }

    static void test_m_n(void(test_func)(uint32_t, uint32_t))
    {
        uint32_t dimss[][2] = {
            {1, 1},
            {10, 10},
            {1, 100},
            {100, 1},
        };
        for (auto dims : dimss) {
            test_func(dims[0], dims[1]);
        }
    }

    static void test_1() { test_k_m_n(test_mm); }
    static void test_2() { test_k_m_n(test_mmt); }
    static void test_3() { test_k_m_n(test_mtm); }
    static void test_4() { test_m_n(test_mv); }
    static void test_5() { test_m_n(test_vm); }
    static void test_6() { test_vv(4); }

    static void test_all()
    {
        test_1();
        test_2();
        test_3();
        test_4();
        test_5();
        test_6();
    }
};

int main()
{
    {
        using T = float;
        using linag = plain_impl<T>;
        test_linag<T, linag>::test_all();
    }
    {
        using T = double;
        using linag = plain_impl<T>;
        test_linag<T, linag>::test_all();
    }
    return 0;
}
