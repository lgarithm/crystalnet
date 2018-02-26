#include <stdint.h> // for uint8_t

#include <misaka.h>

void test_1()
{
    shape_t *shape = make_shape(4, 2, 3, 4, 5);
    tensor_t *tensor = new_tensor(shape, dtypes.f32);
    free_tensor(tensor);
    free_shape(shape);
}

int main()
{
    test_1();
    return 0;
}
