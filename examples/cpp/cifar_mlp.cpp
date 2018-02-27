#include <misaka/misaka>

// fc(w, b)(x) = xw + b
// fc'(w, b)(x) = relu(fc(x))
// y = softmax \circ fc(w3, b3) \circ fc'(w2, b2) \circ fc'(w1 ,b1)
model_t *mlp_model(const shape_t *image_shape, uint32_t arity)
{
    auto l1 = fc_layer(128);
    auto l2 = fc_layer(64);
    auto l3 = fc_layer(arity);
    auto act = relu_layer();
    auto act_out = softmax_layer();
    chain_layer_t chain(&l1, &act, &l2, &act, &l3, &act_out);
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
