#include <misaka/misaka>

// fc(w, b)(x) = xw + b
// fc'(w, b)(x) = relu(fc(x))
// y = softmax \circ fc(w3, b3) \circ fc'(w2, b2) \circ fc'(w1 ,b1)
model_t *mlp_model(shape_t *image_shape, uint32_t arity)
{
    const uint32_t n0 = shape_dim(image_shape);
    const uint32_t n1 = 128;
    const uint32_t n2 = 64;
    const uint32_t n3 = arity;

    model_ctx_t *m = new_model_ctx();
    auto x_ = place(m, *image_shape);
    auto x_wrap_shape = shape(n0);
    auto x = wrap(m, &x_wrap_shape, x_);

    using T = float;
    const truncated_normal_initializer<T> weight_init(0.1);
    const constant_initializer<T> bias_init(0.1);

    auto w1 = var(m, shape(n0, n1), weight_init);
    auto b1 = var(m, shape(n1), bias_init);
    auto w2 = var(m, shape(n1, n2), weight_init);
    auto b2 = var(m, shape(n2), bias_init);
    auto w3 = var(m, shape(n2, n3), weight_init);
    auto b3 = var(m, shape(n3), bias_init);

    // layer 1
    auto op1_1 = apply(m, op_mul, x, w1);
    auto op1_2 = apply(m, op_add, op1_1, b1);
    auto op1_3 = apply(m, op_relu, op1_2);
    // layer 2
    auto op2_1 = apply(m, op_mul, op1_3, w2);
    auto op2_2 = apply(m, op_add, op2_1, b2);
    auto op2_3 = apply(m, op_relu, op2_2);
    // layer 3
    auto op3_1 = apply(m, op_mul, op2_3, w3);
    auto op3_2 = apply(m, op_add, op3_1, b3);
    auto op3_3 = apply(m, op_softmax, op3_2);

    return new_model(m, x_, op3_3);
}

int main()
{
    int width = 28;
    int height = 28;
    int depth = 1;
    int n = 10;
    auto image_shape = shape(width, height, depth);
    model_t *model = mlp_model(&image_shape, n);
    trainer_t *trainer = new_trainer(model, op_xentropy, opt_adam);
    dataset_t *ds1 = load_mnist("train");
    dataset_t *ds2 = load_mnist("t10k");
    // run_trainer(trainer, ds1);
    // test_trainer(trainer, ds2);
    experiment(trainer, ds1, ds2);
    free_model(model);
    free_trainer(trainer);
    free_dataset(ds1);
    free_dataset(ds2);
    return 0;
}
