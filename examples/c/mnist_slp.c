#include <stdint.h>

#include <crystalnet.h>

// y = softmax(flatten(x) * w + b)
s_model_t *slp(const shape_t *image_shape, uint32_t arity)
{
    shape_ctx_t *sc = new_shape_ctx();
    s_model_ctx_t *ctx = make_s_model_ctx();

    symbol x = var(ctx, image_shape);
    symbol x_ = reshape(ctx, mk_shape(sc, 1, shape_dim(image_shape)), x);
    symbol w = covar(ctx, mk_shape(sc, 2, shape_dim(image_shape), arity));
    symbol b = covar(ctx, mk_shape(sc, 1, arity));

    symbol op1 = apply(ctx, op_mul, (symbol[]){x_, w});
    symbol op2 = apply(ctx, op_add, (symbol[]){op1, b});
    symbol op3 = apply(ctx, op_softmax, (symbol[]){op2});

    del_shape_ctx(sc);
    return new_s_model(ctx, x, op3);
}

int main()
{
    const uint32_t width = 28;
    const uint32_t height = 28;
    const uint32_t n = 10;
    const shape_t *image_shape = new_shape(2, width, height);
    s_model_t *model = slp(image_shape, n);
    s_trainer_t *trainer = new_s_trainer(model, op_xentropy, opt_sgd);
    dataset_t *ds1 = load_mnist("train");
    dataset_t *ds2 = load_mnist("t10k");
    s_experiment(trainer, ds1, ds2, 10000);
    del_dataset(ds1);
    del_dataset(ds2);
    del_s_trainer(trainer);
    del_s_model(model);
    del_shape(image_shape);
    return 0;
}
