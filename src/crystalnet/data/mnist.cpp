#include <cstdio>
#include <cstdlib>
#include <memory>

#include <crystalnet.h>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/data/dataset.hpp>

tensor_t *_load_mnist(const std::string &name)
{
    const auto filename = data_dir() + "/mnist/" + name;
    return _load_idx_file(filename.c_str());
}

dataset_t *load_mnist_data(const std::string &name)
{
    auto images =
        std::unique_ptr<tensor_t>(_load_mnist(name + "-images-idx3-ubyte"));
    tensor_t *images_ = cast_to<float>(r_tensor_ref_t<uint8_t>(*images));
    normalize<float>(r_tensor_ref_t<float>(*images_), 255.0);
    auto labels =
        std::unique_ptr<tensor_t>(_load_mnist(name + "-labels-idx1-ubyte"));
    tensor_t *labels_ = make_onehot<float>(*labels, 10);
    return new simple_dataset_t(images_, labels_);
}

dataset_t *load_mnist(const char *const name) { return load_mnist_data(name); }
