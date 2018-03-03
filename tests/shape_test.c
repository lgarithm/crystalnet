#include <assert.h>
#include <stdio.h>

#include <crystalnet.h>

void test_1()
{
    {
        shape_t *shape = make_shape(0);
        assert(shape_dim(shape) == 1);
        assert(shape_rank(shape) == 0);
        free_shape(shape);
    }
    {
        shape_t *shape = make_shape(4, 2, 3, 4, 5);
        assert(shape_dim(shape) == 120);
        assert(shape_rank(shape) == 4);
        free_shape(shape);
    }
}

int main()
{
    test_1();
    return 0;
}
