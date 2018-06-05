#pragma once
#include <stdint.h>

#include <crystalnet-ext.h>

#ifdef __cplusplus
extern "C" {
#endif

const uint32_t vgg16_image_size = 224;
const uint32_t vgg16_class_number = 1000;

// https://www.cs.toronto.edu/~frossard/post/vgg16/
s_model_t *vgg16(context_t *, const shape_t *, uint32_t);

#ifdef __cplusplus
}
#endif
