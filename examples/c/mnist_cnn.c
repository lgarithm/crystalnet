#include <stddef.h>
#include <stdint.h>

#include <crystalnet-ext.h>

// l1 = pool(relu(conv(x)))
// l2 = pool(relu(conv(l1)))
// y = softmax(dense(relu(dense(l2))))

typedef shape_t const *p_shape_t;

s_model_t *cnn(context_t *ctx, const shape_t *image_shape, uint32_t arity)
{
    s_layer_t *c1 = new_layer_conv_nhwc(5, 5, 32);
    s_layer_t *c2 = new_layer_conv_nhwc(5, 5, 64);
    s_layer_t *f1 = new_layer_dense(1024);
    s_layer_t *f2 = new_layer_dense(arity);
    s_layer_t *pool = new_layer_pool_max();  //
    s_layer_t *act = new_layer_relu();       //
    s_layer_t *out = new_layer_softmax();

    symbol x = var(ctx, image_shape);
    symbol x_ = reshape(ctx, mk_shape(ctx, 3, 28, 28, 1), x);
    symbol y = transform_all(ctx,
                             (p_layer_t[]){
                                 c1,
                                 act,
                                 pool,
                                 c2,
                                 act,
                                 pool,
                                 f1,
                                 act,
                                 f2,
                                 out,
                                 NULL,
                             },
                             x_);
    del_s_layer(c1);
    del_s_layer(c2);
    del_s_layer(f1);
    del_s_layer(f2);
    del_s_layer(pool);
    del_s_layer(act);
    del_s_layer(out);
    return make_s_model(ctx, x, y);
}

int main()
{
    const uint32_t height = 28;
    const uint32_t width = 28;
    const uint32_t n = 10;
    context_t *ctx = new_context();
    const shape_t *image_shape = mk_shape(ctx, 2, height, width);
    s_model_t *model = cnn(ctx, image_shape, n);
    s_trainer_t *trainer = new_s_trainer(model, op_xentropy, opt_adam);
    dataset_t *ds1 = load_mnist("train");
    dataset_t *ds2 = load_mnist("t10k");
    s_experiment(trainer, ds1, ds2, 10);
    del_dataset(ds1);
    del_dataset(ds2);
    del_s_trainer(trainer);
    del_context(ctx);
    return 0;
}
