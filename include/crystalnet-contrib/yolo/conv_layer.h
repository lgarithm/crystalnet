#pragma once

#include <crystalnet-ext.h>

#ifdef __cplusplus
extern "C" {
#endif

extern s_layer_t *conv(uint32_t /* filters */, uint32_t /* size */,
                       uint32_t /* stride */, uint32_t /* padding */);

extern s_layer_t *conv_linear_act(uint32_t /* filters */, uint32_t /* size */,
                                  uint32_t /* stride */,
                                  uint32_t /* padding */);

#ifdef __cplusplus
}
#endif
