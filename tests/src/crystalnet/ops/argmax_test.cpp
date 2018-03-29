#include <cassert>

#include <crystalnet/ops/argmax.hpp>

void test_1()
{
    using T = float;
    tensor_t _values(shape_t(100), idx_type<T>::type);
    tensor_t _tops(shape_t(3), idx_type<int32_t>::type);

    const auto values = ranked<1, T>(ref(_values));
    const auto tops = ranked<1, int32_t>(ref(_tops));

    values.data[10] = 100.0;
    values.data[20] = 200.0;
    values.data[30] = 300.0;

    top_indexes(values, tops);

    assert(tops.data[0] == 30);
    assert(tops.data[1] == 20);
    assert(tops.data[2] == 10);
}

int main()
{
    test_1();
    return 0;
}
