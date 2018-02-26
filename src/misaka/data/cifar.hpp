#pragma once

#include <cstdint>
#include <memory>

#include <misaka.h>
#include <misaka/core/idx.hpp>
#include <misaka/core/shape.hpp>
#include <misaka/core/tensor.hpp>
#include <misaka/data/cifar.hpp>
#include <misaka/data/dataset.hpp>

dataset_t *load_cifar10_data(const std::string &filename)
{
    constexpr uint32_t n = 10000;
    // TODO: use std::make_unique
    // Undefined symbols for architecture x86_64:
    //   "idx_type<unsigned char>::type", referenced from:
    //       load_cifar10_data(std::__1::basic_string<char,
    //       std::__1::char_traits<char>, std::__1::allocator<char> > const&) in
    //       cifar.cpp.o
    // auto labels = std::make_unique<tensor_t>(shape_t(n),
    // idx_type<uint8_t>::type);
    auto labels = std::unique_ptr<tensor_t>(
        new tensor_t(shape_t(n), idx_type<uint8_t>::type));
    auto images = std::unique_ptr<tensor_t>(
        new tensor_t(shape_t(n, 3, 32, 32), idx_type<uint8_t>::type));
    FILE *fp = fopen(filename.c_str(), "r");
    for (auto i = 0; i < n; ++i) {
        fread((char *)labels->data + i, 1, 1, fp);
        fread((char *)images->data + i * 32 * 32 * 3, 3, 32 * 32, fp);
    }
    fclose(fp);
    tensor_t *images_ = cast_to<float>(r_tensor_ref_t<uint8_t>(*images));
    normalize<float>(r_tensor_ref_t<float>(*images_), 255.0);
    return new simple_dataset_t(images_, make_onehot<float>(*labels, 10));
}

dataset_t *load_cifar()
{
    // data_batch_{1,2,3,4,5}.bin
    // test_batch.bin
    const auto name = "test_batch.bin";
    const auto filename = data_dir() + "/cifar/cifar-10-batches-bin/" + name;
    return load_cifar10_data(filename);
}
