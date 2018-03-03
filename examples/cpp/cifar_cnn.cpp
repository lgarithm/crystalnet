#include <misaka/misaka>

// l1 = pool \circ relu \circ conv1
// l2 = pool \circ relu \circ conv2
// y = softmax \circ fc \circ l2 \circ l2
model_t *mlp_model(const shape_t *image_shape, uint32_t arity)
{
    auto c1 = conv_layer(5, 5, 32);
    auto c2 = conv_layer(5, 5, 64);
    auto f3 = fc_layer(1024);
    auto f4 = fc_layer(arity);
    auto pool = pool_layer();
    auto act = relu_layer();
    auto act_out = softmax_layer();
    chain_layer_t chain(&c1, &act, &pool, //
                        &c2, &act, &pool, //
                        &f3, &act, &f4,   //
                        &act_out);
    model_ctx_t *ctx = new_model_ctx();
    auto x = place(ctx, *image_shape);
    auto y = chain(*ctx, x);
    return new_model(ctx, x, y);
}

int main()
{
    dataset_t *ds1 = load_cifar();
    dataset_t *ds2 = load_cifar(); // TODO: load cifar test data
    auto image_shape = ds_image_shape(ds1);
    auto label_shape = ds_label_shape(ds1);
    model_t *model = mlp_model(image_shape, shape_dim(label_shape));
    trainer_t *trainer = new_trainer(model, op_xentropy, opt_adam);
    experiment(trainer, ds1, ds2);
    free_model(model);
    free_trainer(trainer);
    free_dataset(ds1);
    free_dataset(ds2);
    return 0;
}
