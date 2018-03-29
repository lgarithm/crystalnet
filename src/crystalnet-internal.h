#pragma once

#include <crystalnet.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct model_t model_t;
typedef struct model_ctx_t model_ctx_t;
typedef struct parameter_ctx_t parameter_ctx_t;

extern model_t *realize(parameter_ctx_t *, const s_model_t *, uint32_t);

// testing APIs
extern parameter_ctx_t *new_parameter_ctx();
extern void del_parameter_ctx(parameter_ctx_t *);
extern void del_model(model_t *);

#ifdef __cplusplus
}
#endif
