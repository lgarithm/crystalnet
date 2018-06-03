// https://github.com/pjreddie/darknet.git
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

extern void reorg_cpu(float *x, int w, int h, int c, int batch, int stride,
                      int forward, float *out);

#ifdef __cplusplus
}
#endif
