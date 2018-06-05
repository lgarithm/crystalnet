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
extern void del_parameter_ctx(const parameter_ctx_t *);
extern void del_model(const model_t *);

// unstable APIs
typedef struct shape_list_t shape_list_t;
extern const shape_list_t *mk_shape_list(context_t *,
                                         const shape_t *const p_shapes[]);

// TODO: make it possible to add user defined operators
typedef struct forward_ctx_t forward_ctx_t;
typedef struct backward_ctx_t backward_ctx_t;
typedef struct shape_func_t shape_func_t;
typedef struct forward_func_t forward_func_t;
typedef struct backward_func_t backward_func_t;
extern const operator_t *register_op(const char *const, uint8_t, shape_func_t *,
                                     forward_func_t *, backward_func_t *);

// eager APIs
extern shape_t *infer(const operator_t *, const shape_list_t *);
extern void eval(const operator_t *, const forward_ctx_t *);
extern void grad(const operator_t *, const backward_ctx_t *);

#ifdef __cplusplus
}
#endif
