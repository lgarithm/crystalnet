#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <crystalnet.h>

tensor_t *_load_mnist(const char *const name)
{
    const char *const prefix = "var/data/mnist";
    char filename[1024];
    sprintf(filename, "%s/%s/%s", getenv("HOME"), prefix, name);
    printf("%s\n", filename);
    tensor_t *t = _load_idx_file(filename);
    return t;
}

void test_1()
{
    {
        tensor_t *t = _load_mnist("t10k-labels-idx1-ubyte");
        const shape_t *s = tensor_shape(t);
        assert(shape_rank(s) == 1);
        assert(shape_dim(s) == 10000);
        del_tensor(t);
    }
    {
        tensor_t *t = _load_mnist("t10k-images-idx3-ubyte");
        const shape_t *s = tensor_shape(t);
        assert(shape_rank(s) == 3);
        assert(shape_dim(s) == 28 * 28 * 10000);
        del_tensor(t);
    }
}

int main()
{
    test_1();
    return 0;
}
