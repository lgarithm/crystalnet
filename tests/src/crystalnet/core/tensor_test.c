#include <stdint.h> // for uint8_t

#include <crystalnet.h>

void test_1()
{
    const shape_t *shape = new_shape(4, 2, 3, 4, 5);
    tensor_t *tensor = new_tensor(shape, dtypes.f32);
    del_tensor(tensor);
    del_shape(shape);
}

int main()
{
    test_1();
    return 0;
}
