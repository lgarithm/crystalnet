#include <misaka/misaka>

// TODO: complete

// y = xw + b
model_ctx_t *slp_model(shape_t *image_shape, int arity, int batch_size)
{
    auto lable_shape = make_shape(1, arity);
    auto m = new_model_ctx();
    auto *x = make_placeholder(m, image_shape);
    auto *y_ = make_placeholder(m, lable_shape);

    //
    // auto l1 =
    //     make_layer(layer_fc_with_bias(k), wrap(shape(m, height * width), x));
    // output = make_operator(batch<op_softmax>(), l1);
    // auto ce = make_operator(xybatch<op_cross_entropy>(), y_, output);
    // loss_ = make_operator(op_sum(), ce);
    free_shape(lable_shape);
    return m;
}

void train(model_ctx_t *model) {}

int main()
{
    int width = 28;
    int height = 28;
    int depth = 1;
    int n = 10;
    int batch_size = 500;
    // struct DataSet ds = load_dataset_mnist();
    // shape_t* image_shape = new_shape(width, height, depth);
    shape_t *image_shape = make_shape(3, width, height, depth);
    model_ctx_t *m = slp_model(image_shape, n, batch_size);
    train(m);
    free_shape(image_shape);
    return 0;
}
