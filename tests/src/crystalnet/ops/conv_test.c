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
                 mk_shape(ctx, 4, 3, 3, 32, 64),
                 NULL,
             });
    const operator_t *op = make_op_conv2d(0, 0, 1, 1);
    shape_t *out_shape = infer(op, shape_list);
    assert(shape_rank(out_shape) == 3);
    assert(shape_dim(out_shape) == 26 * 26 * 64);
    del_shape(out_shape);
    del_context(ctx);
}

void test_2()
{
    context_t *ctx = new_context();
    const shape_list_t *shape_list = mk_shape_list(  //
        ctx, (p_shape_t[]){
                 mk_shape(ctx, 4, 2, 28, 28, 32),
                 mk_shape(ctx, 4, 3, 3, 32, 64),
                 NULL,
             });
    const operator_t *op = make_op_conv2d(0, 0, 1, 1);
    shape_t *out_shape = infer(op, shape_list);
    assert(shape_rank(out_shape) == 4);
    assert(shape_dim(out_shape) == 2 * 26 * 26 * 64);
    del_shape(out_shape);
    del_context(ctx);
}

int main()
{
    test_1();
    test_2();
    return 0;
}
