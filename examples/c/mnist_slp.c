#include <stdio.h>

#include <misaka.h>

// y = softmax(xw + b)
model_t *slp_model(shape_t *image_shape, int arity)
{
    shape_t *lable_shape = make_shape(1, arity);
    shape_t *weight_shape = make_shape(2, shape_dim(image_shape), arity);
    shape_t *x_wrap_shape = make_shape(1, shape_dim(image_shape));

    model_ctx_t *m = new_model_ctx();
    node_t *x_ = make_placeholder(m, image_shape);
    node_t *x = wrap(m, x_wrap_shape, x_);
    node_t *w = make_parameter(m, weight_shape);
    node_t *b = make_parameter(m, lable_shape);

    node_t *args1[] = {x, w};
    node_t *op1 = make_operator(m, op_mul, args1);
    node_t *args2[] = {op1, b};
    node_t *op2 = make_operator(m, op_add, args2);

    node_t *args3[] = {op2};
    node_t *op3 = make_operator(m, op_softmax, args3);

    free_shape(lable_shape);
    free_shape(weight_shape);
    free_shape(x_wrap_shape);
    return new_model(m, x_, op3);
}

model_ctx_t *slp_model_2(shape_t *image_shape, int arity)
{
    // layer_t* fc1 = make_layer(m, Layer_FC, x);
    // placeholder_t *y_ = make_placeholder(m, lable_shape);
    //
    // auto l1 =
    //     make_layer(layer_fc_with_bias(k), wrap(shape(m, height * width), x));
    // output = make_operator(batch<op_softmax>(), l1);
    // auto ce = make_operator(xybatch<op_cross_entropy>(), y_, output);
    model_ctx_t *m = new_model_ctx();
    return m;
}

void show_version() { printf("misaka: %s\n", version()); }

int main()
{
    show_version();
    int width = 28;
    int height = 28;
    int depth = 1;
    int n = 10;
    shape_t *image_shape = make_shape(3, width, height, depth);
    model_t *model = slp_model(image_shape, n);
    trainer_t *trainer = new_trainer(model, op_xentropy, opt_sgd);
    dataset_t *ds1 = load_mnist();
    dataset_t *ds2 = load_mnist(); // TODO: load test data
    run_trainer(trainer, ds1);
    test_trainer(trainer, ds2);
    free_shape(image_shape);
    free_model(model);
    free_trainer(trainer);
    free_dataset(ds1);
    free_dataset(ds2);
    return 0;
}
