#include <assert.h>

#include <crystalnet.h>

// y = softmax(flatten(x) * w + b)
s_model_t *slp(shape_t *image_shape, uint8_t arity)
{
    shape_t *lable_shape = make_shape(1, arity);
    shape_t *weight_shape = make_shape(2, shape_dim(image_shape), arity);
    shape_t *x_wrap_shape = make_shape(1, shape_dim(image_shape));

    s_model_ctx_t *ctx = new_s_model_ctx();
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

    free_shape(lable_shape);
    free_shape(weight_shape);
    return new_s_model(ctx, x, op3);
}

void test_1()
{
    uint8_t arity = 10;
    shape_t *image_shape = make_shape(2, 28, 28);
    s_model_t *model = slp(image_shape, arity);
    free_s_model(model);
    free_shape(image_shape);
}

int main()
{
    test_1();
    return 0;
}
