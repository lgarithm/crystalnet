#pragma once
#include <cstdio>
#include <cstdlib>

#include <misaka.h>
#include <misaka/core/tensor.hpp>
#include <misaka/data/dataset.hpp>

tensor_t *_load_mnist(const char *const name)
{
    DEBUG(__func__);
    const char *const prefix = "var/data/mnist";
    char filename[1024];
    sprintf(filename, "%s/%s/%s", getenv("HOME"), prefix, name);
    return _load_idx_file(filename);
}

template <typename T> tensor_t *make_onehot(const tensor_t &tensor, uint32_t k)
{
    auto dims = std::vector<uint32_t>(tensor.shape.dims);
    dims.push_back(k);
    auto distro_ = new tensor_t(shape_t(dims), idx_type<T>::type);
    r_tensor_ref_t<T> distro(*distro_);
    r_tensor_ref_t<uint8_t> r(tensor); // TODO: support other uint types
    auto n = tensor.shape.dim();
    for (auto i = 0; i < n; ++i) {
        auto off = r.data[i];
        if (0 <= off && off < k) {
            distro.data[i * k + off] = 1;
        } else {
            // TODO: print a warning msg
            assert(false);
        }
    }
    return distro_;
}

dataset_t *load_mnist_data(const char *name)
{
    DEBUG(__func__);
    tensor_t *images = _load_mnist("train-images-idx3-ubyte");
    tensor_t *images_ = cast_to<float>(r_tensor_ref_t<uint8_t>(*images));
    delete images;
    {
        r_tensor_ref_t<float> r(*images_);
        auto n = r.shape.dim();
        for (auto i = 0; i < n; ++i) {
            r.data[i] /= 255.0;
        }
    }
    tensor_t *labels = _load_mnist("train-labels-idx1-ubyte");
    tensor_t *labels_ = make_onehot<float>(*labels, 10);
    delete labels;
    return new simple_dataset_t(images_, labels_);
}

dataset_t *load_mnist()
{
    DEBUG(__func__);
    return load_mnist_data("train");
}
