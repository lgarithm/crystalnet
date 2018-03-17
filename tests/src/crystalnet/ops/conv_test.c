#include <assert.h>
#include <stdio.h>

#include <crystalnet.h>

typedef shape_t const *p_shape_t;

void test_1()
{
    shape_ctx_t *sc = make_shape_ctx();
    const shape_list_t *shape_list = mk_shape_list( //
        sc, (p_shape_t[]){
                mk_shape(sc, 3, 28, 28, 32),
                mk_shape(sc, 4, 3, 3, 32, 64),
                NULL,
            });
    operator_t *op = make_op_conv2d(0, 0, 1, 1);
    shape_t *out_shape = infer(op, shape_list);
    assert(shape_rank(out_shape) == 3);
    assert(shape_dim(out_shape) == 26 * 26 * 64);
    free_shape(out_shape);
    free_shape_ctx(sc);
}

void test_2()
{
    shape_ctx_t *sc = make_shape_ctx();
    const shape_list_t *shape_list = mk_shape_list( //
        sc, (p_shape_t[]){
                mk_shape(sc, 4, 2, 28, 28, 32),
                mk_shape(sc, 4, 3, 3, 32, 64),
                NULL,
            });
    operator_t *op = make_op_conv2d(0, 0, 1, 1);
    shape_t *out_shape = infer(op, shape_list);
    assert(shape_rank(out_shape) == 4);
    assert(shape_dim(out_shape) == 2 * 26 * 26 * 64);
    free_shape(out_shape);
    free_shape_ctx(sc);
}

int main()
{
    test_1();
    test_2();
    return 0;
}
