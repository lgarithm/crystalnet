#include <crystalnet.h>

// y = softmax(flatten(x) * w + b)
s_model_t *slp(context_t *ctx, const shape_t *image_shape, uint8_t arity)
{
    symbol x = var(ctx, image_shape);
    symbol x_ = reshape(ctx, mk_shape(ctx, 1, shape_dim(image_shape)), x);
    symbol w = covar(ctx, mk_shape(ctx, 2, shape_dim(image_shape), arity));
    symbol b = covar(ctx, mk_shape(ctx, 1, arity));

    symbol op1 = apply(ctx, op_mul, (symbol[]){x_, w});
    symbol op2 = apply(ctx, op_add, (symbol[]){op1, b});
    symbol op3 = apply(ctx, op_softmax, (symbol[]){op2});

    return make_s_model(ctx, x, op3);
}

int main()
{
    int width = 32;
    int height = 32;
    int depth = 3;
    int n = 10;
    context_t *ctx = new_context();
    const shape_t *image_shape = mk_shape(ctx, 3, depth, width, height);
    s_model_t *model = slp(ctx, image_shape, n);
    s_trainer_t *trainer = new_s_trainer(model, op_xentropy, opt_sgd);
    dataset_t *ds1 = load_cifar();
    dataset_t *ds2 = load_cifar();
    s_experiment(trainer, ds1, ds2, 10000);
    del_s_trainer(trainer);
    del_dataset(ds1);
    del_dataset(ds2);
    del_context(ctx);
    return 0;
}
