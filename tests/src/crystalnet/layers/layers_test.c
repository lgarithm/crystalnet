#include <crystalnet-ext.h>

void test_1()
{
    s_layer_t *l1 = new_layer_dense(10);
    del_s_layer(l1);
}

int main()
{
    test_1();
    return 0;
}
