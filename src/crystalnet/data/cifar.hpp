#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>

#include <crystalnet.h>
#include <crystalnet/core/idx.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/data/cifar.hpp>
#include <crystalnet/data/dataset.hpp>

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
        new tensor_t(shape_t(n, 32, 32, 3), idx_type<uint8_t>::type));
    uint8_t buffer[32 * 32 * 3];
    FILE *fp = fopen(filename.c_str(), "r");
    uint8_t *p = (uint8_t *)images->data;
    for (auto l = 0; l < n; ++l) {
        check(std::fread((char *)labels->data + l, 1, 1, fp) == 1);
        check(std::fread((char *)buffer, 3, 32 * 32, fp) == 32 * 32);
        // [c, n, n] -> [n, n, c]
        // fread((char *)images->data + i * 32 * 32 * 3, 3, 32 * 32, fp);
        for (auto i = 0; i < 32; ++i) {
            for (auto j = 0; j < 32; ++j) {
                for (auto k = 0; k < 3; ++k) {
                    // buffer[k, i, j]
                    *p++ = buffer[(k * 32 + i) * 32 + j];
                }
            }
        }
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
