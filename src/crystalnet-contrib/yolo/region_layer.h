#pragma once

#include <crystalnet-ext.h>

#ifdef __cplusplus
extern "C" {
#endif

s_layer_t *make_region_layer(int w, int h, int n, int classes, int coords);

#ifdef __cplusplus
}
#endif
