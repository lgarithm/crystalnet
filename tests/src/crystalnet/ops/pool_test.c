#include <assert.h>
#include <stdio.h>

#include <crystalnet-internal.h>

typedef shape_t const *p_shape_t;

void test_1()
{
    context_t *ctx = new_context();
    const shape_list_t *shape_list = mk_shape_list(  //
        ctx, (p_shape_t[]){
                 mk_shape(ctx, 3, 28, 28, 32),
                 NULL,
             });
    const operator_t *op = make_op_pool2d(2, 2, 2, 2);
    shape_t *out_shape = infer(op, shape_list);
    assert(shape_rank(out_shape) == 3);
    assert(shape_dim(out_shape) == 14 * 14 * 32);
    del_shape(out_shape);
    del_context(ctx);
}

void test_2()
{
    context_t *ctx = new_context();
    const shape_list_t *shape_list = mk_shape_list(  //
        ctx, (p_shape_t[]){
                 mk_shape(ctx, 4, 10, 28, 28, 32),
                 NULL,
             });
    const operator_t *op = make_op_pool2d(2, 2, 2, 2);
    shape_t *out_shape = infer(op, shape_list);
    assert(shape_rank(out_shape) == 4);
    assert(shape_dim(out_shape) == 10 * 14 * 14 * 32);
    del_shape(out_shape);
    del_context(ctx);
}

int main()
{
    test_1();
    test_2();
    return 0;
}
