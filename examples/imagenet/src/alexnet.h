#pragma once
#include <stdint.h>

#include <crystalnet-ext.h>

#ifdef __cplusplus
extern "C" {
#endif

const uint32_t alexnet_image_size = 227;
const uint32_t alexnet_class_number = 1000;

// https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
s_model_t *alexnet(const shape_t *, uint32_t);

#ifdef __cplusplus
}
#endif
