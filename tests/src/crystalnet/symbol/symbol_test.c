#include <assert.h>

#include <crystalnet-internal.h>

// y = softmax(flatten(x) * w + b)
s_model_t *slp(context_t *ctx, const shape_t *image_shape, uint8_t arity)
{
    const shape_t *lable_shape = new_shape(1, arity);
    const shape_t *weight_shape = new_shape(2, shape_dim(image_shape), arity);
    const shape_t *x_wrap_shape = new_shape(1, shape_dim(image_shape));

    s_node_t *x = var(ctx, image_shape);
    s_node_t *x_ = reshape(ctx, x_wrap_shape, x);
    s_node_t *w = covar(ctx, weight_shape);
    s_node_t *b = covar(ctx, lable_shape);

    s_node_t *args1[] = {x_, w};
    s_node_t *op1 = apply(ctx, op_mul, args1);
    s_node_t *args2[] = {op1, b};
    s_node_t *op2 = apply(ctx, op_add, args2);
    s_node_t *args3[] = {op2};
    s_node_t *op3 = apply(ctx, op_softmax, args3);

    del_shape(lable_shape);
    del_shape(weight_shape);
    del_shape(x_wrap_shape);
    return make_s_model(ctx, x, op3);
}

void test_1()
{
    uint8_t arity = 10;
    const shape_t *image_shape = new_shape(2, 28, 28);
    context_t *ctx = new_context();
    s_model_t *sm = slp(ctx, image_shape, arity);
    parameter_ctx_t *pc = new_parameter_ctx();
    for (int i = 1; i <= 3; ++i) {
        model_t *pm = realize(pc, sm, i);
        del_model(pm);
    }
    del_parameter_ctx(pc);
    del_context(ctx);
    del_shape(image_shape);
}

int main()
{
    test_1();
    return 0;
}
