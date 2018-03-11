#include <crystalnet/ops/softmax.hpp>

void test_1()
{
    const auto shape = shape_t(999);
    tensor_t x(shape);
    tensor_t y(shape);
    forward_ctx_t ctx(tensor_ref_list_t(std::vector<tensor_ref_t>({ref(x)})),
                      ref(y));
    call<softmax::forward>(ctx);
}

void test_2()
{
    const auto shape = shape_t(999, 10);
    tensor_t x(shape);
    tensor_t y(shape);
    forward_ctx_t ctx(tensor_ref_list_t(std::vector<tensor_ref_t>({ref(x)})),
                      ref(y));
    call<softmax::forward>(ctx);
}

int main()
{
    test_1();
    test_2();
    return 0;
}