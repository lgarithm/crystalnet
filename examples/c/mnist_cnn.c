#include <assert.h>
#include <stdio.h>

#include <crystalnet.h>

// l1 = pool(relu(conv(x)))
// l2 = pool(relu(conv(l1)))
// y = softmax(dense(relu(dense(l2))))

typedef s_layer_t const *p_layer_t;

s_node_t *transform_all(s_model_ctx_t *ctx, p_layer_t ls[], s_node_t *x)
{
    for (p_layer_t *pl = ls; *pl; ++pl) {
        x = transform(ctx, *pl, x);
    }
    return x;
}

typedef shape_t const *p_shape_t;
s_model_t *cnn(shape_t *image_shape, uint8_t arity)
{
    shape_ctx_t *sc = make_shape_ctx();
    s_model_ctx_t *ctx = new_s_model_ctx();

    s_layer_t *c1 =
        new_layer_conv_nhwc(mk_shape_list(sc, (p_shape_t[]){
                                                  mk_shape(sc, 3, 5, 5, 32),
                                                  NULL,
                                              }));
    s_layer_t *c2 =
        new_layer_conv_nhwc(mk_shape_list(sc, (p_shape_t[]){
                                                  mk_shape(sc, 3, 5, 5, 64),
                                                  NULL,
                                              }));
    s_layer_t *f1 = new_layer_dense(mk_shape_list(sc, (p_shape_t[]){
                                                          mk_shape(sc, 1, 1024),
                                                          NULL,
                                                      }));
    s_layer_t *f2 =
        new_layer_dense(mk_shape_list(sc, (p_shape_t[]){
                                              mk_shape(sc, 1, arity),
                                              NULL,
                                          }));
    s_layer_t *pool = new_layer_pool_max(NULL); //
    s_layer_t *act = new_layer_relu(NULL);      //
    s_layer_t *out = new_layer_softmax(NULL);

    printf("[x] creating model\n");
    symbol x = var(ctx, image_shape);
    symbol x_ = reshape(ctx, mk_shape(sc, 3, 28, 28, 1), x);
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
    free_shape_ctx(sc);
    free_s_layer(c1);
    free_s_layer(c2);
    free_s_layer(f1);
    free_s_layer(f2);
    free_s_layer(pool);
    free_s_layer(act);
    free_s_layer(out);
    printf("[y] creating model\n");
    return new_s_model(ctx, x, y);
}

int main()
{
    const uint32_t height = 28;
    const uint32_t width = 28;
    const uint32_t n = 10;
    shape_t *image_shape = make_shape(2, height, width);
    s_model_t *model = cnn(image_shape, n);
    s_trainer_t *trainer = new_s_trainer(model, op_xentropy, opt_adam);
    dataset_t *ds1 = load_mnist("train");
    dataset_t *ds2 = load_mnist("t10k");
    s_experiment(trainer, ds1, ds2, 10);
    free_dataset(ds1);
    free_dataset(ds2);
    free_s_model(model);
    free_s_trainer(trainer);
    free_shape(image_shape);
    return 0;
}
