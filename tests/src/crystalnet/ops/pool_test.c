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
                NULL,
            });
    operator_t *op = make_op_pool2d(2, 2, 2, 2);
    shape_t *out_shape = infer(op, shape_list);
    assert(shape_rank(out_shape) == 3);
    assert(shape_dim(out_shape) == 14 * 14 * 32);
    free_shape(out_shape);
    free_shape_ctx(sc);
}

void test_2()
{
    shape_ctx_t *sc = make_shape_ctx();
    const shape_list_t *shape_list = mk_shape_list( //
        sc, (p_shape_t[]){
                mk_shape(sc, 4, 10, 28, 28, 32),
                NULL,
            });
    operator_t *op = make_op_pool2d(2, 2, 2, 2);
    shape_t *out_shape = infer(op, shape_list);
    assert(shape_rank(out_shape) == 4);
    assert(shape_dim(out_shape) == 10 * 14 * 14 * 32);
    free_shape(out_shape);
    free_shape_ctx(sc);
}

int main()
{
    test_1();
    test_2();
    return 0;
}
