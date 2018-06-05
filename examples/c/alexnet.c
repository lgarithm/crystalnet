#include <stdio.h>

#include <crystalnet-ext.h>

typedef shape_t const *p_shape_t;

// https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
s_model_t *alexnet(context_t *ctx, const shape_t *image_shape, uint32_t arity)
{
    s_layer_t *c1 = new_layer_conv2d(           //
        mk_shape(ctx, 3, 11, 11, 96),           //
        NULL,                                   // default padding
        mk_shape(ctx, 2, 4, 4));                //
    s_layer_t *c2 = new_layer_conv2d(           //
        mk_shape(ctx, 3, 5, 5, 256),            //
        mk_shape(ctx, 2, 2, 2),                 //
        NULL);                                  // default stride
    s_layer_t *c3_c4 = new_layer_conv2d(        //
        mk_shape(ctx, 3, 3, 3, 384),            //
        mk_shape(ctx, 2, 1, 1),                 //
        NULL);                                  //
    s_layer_t *c5 = new_layer_conv2d(           //
        mk_shape(ctx, 3, 3, 3, 256),            //
        mk_shape(ctx, 2, 1, 1),                 //
        NULL);                                  //
    s_layer_t *f4096 = new_layer_dense(4096);   //
    s_layer_t *f_out = new_layer_dense(arity);  //
    s_layer_t *pool = new_layer_pool2d(         //
        mk_shape(ctx, 2, 3, 3),                 //
        mk_shape(ctx, 2, 2, 2));                //
    s_layer_t *relu = new_layer_relu();         //
    s_layer_t *out = new_layer_softmax();

    printf("[x] creating model\n");
    symbol x = var(ctx, image_shape);
    symbol y = transform_all(  //
        ctx,                   //
        (p_layer_t[]){
            c1,    relu, pool,                         //
            c2,    relu, pool,                         //
            c3_c4, relu, c3_c4, relu, c5, relu, pool,  //
            f4096, relu, f4096, relu,                  //
            f_out, out,                                //
            NULL,                                      //
        },
        x);
    del_s_layer(c1);
    del_s_layer(c2);
    del_s_layer(c3_c4);
    del_s_layer(c5);
    del_s_layer(f4096);
    del_s_layer(f_out);
    del_s_layer(pool);
    del_s_layer(relu);
    del_s_layer(out);
    printf("[y] creating model\n");
    return make_s_model(ctx, x, y);
}

const uint32_t height = 227;
const uint32_t width = 227;
const uint32_t class_number = 1000;

dataset_t *fake_imagenet()
{
    const shape_t *image_shape = new_shape(3, height, width, 3);
    dataset_t *p_ds = new_fake_dataset(image_shape, class_number);
    del_shape(image_shape);
    return p_ds;
}

int main()
{
    context_t *ctx = new_context();
    const shape_t *image_shape = mk_shape(ctx, 3, height, width, 3);
    s_model_t *model = alexnet(ctx, image_shape, class_number);
    s_model_info(model);
    s_trainer_t *trainer = new_s_trainer(model, op_xentropy, opt_adam);
    dataset_t *ds1 = fake_imagenet();
    // dataset_t *ds2 = fake_imagenet();
    const uint32_t batch_size = 2;
    s_experiment(trainer, ds1, NULL, batch_size);
    del_dataset(ds1);
    // del_dataset(ds2);
    del_s_trainer(trainer);
    del_context(ctx);
    return 0;
}
