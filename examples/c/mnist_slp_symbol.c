#include <assert.h>

#include <crystalnet.h>

// y = softmax(flatten(x) * w + b)
s_model_t *slp(shape_t *image_shape, uint8_t arity)
{
    shape_ctx_t *sc = make_shape_ctx();
    s_model_ctx_t *ctx = new_s_model_ctx();

    symbol x = var(ctx, image_shape);
    symbol x_ = reshape(ctx, mk_shape(sc, 1, shape_dim(image_shape)), x);
    symbol w = covar(ctx, mk_shape(sc, 2, shape_dim(image_shape), arity));
    symbol b = covar(ctx, mk_shape(sc, 1, arity));

    symbol args1[] = {x_, w};
    symbol op1 = apply(ctx, op_mul, args1);
    symbol args2[] = {op1, b};
    symbol op2 = apply(ctx, op_add, args2);
    symbol args3[] = {op2};
    symbol op3 = apply(ctx, op_softmax, args3);

    free_shape_ctx(sc);
    return new_s_model(ctx, x, op3);
}

int main()
{
    int width = 28;
    int height = 28;
    uint8_t n = 10;
    shape_t *image_shape = make_shape(2, width, height);
    s_model_t *sm = slp(image_shape, n);
    model_t *pm = realize(sm);
    trainer_t *trainer = new_trainer(pm, op_xentropy, opt_sgd);
    dataset_t *ds1 = load_mnist("train");
    dataset_t *ds2 = load_mnist("t10k");
    experiment(trainer, ds1, ds2);
    free_dataset(ds1);
    free_dataset(ds2);
    free_model(pm);
    free_s_model(sm);
    free_shape(image_shape);
    return 0;
}
