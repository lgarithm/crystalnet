#pragma once
#include <stdint.h>

#include <crystalnet.h>

#ifdef __cplusplus
extern "C" {
#endif

// layer APIs
typedef struct s_layer_t s_layer_t;
extern void del_s_layer(s_layer_t *);
extern s_layer_t *const new_layer_dense(uint32_t /* n */);
extern s_layer_t *const new_layer_relu();
extern s_layer_t *const new_layer_softmax();
extern s_layer_t *const new_layer_pool2d(const shape_t * /* filter shape */,
                                         const shape_t * /* stride shape */);
extern s_layer_t *const new_layer_conv2d(const shape_t * /* filter shape */,
                                         const shape_t * /* padding shape */,
                                         const shape_t * /* stride shape */);

// TODO: deprecate
extern s_layer_t *const new_layer_conv_nhwc(uint32_t, uint32_t, uint32_t);
extern s_layer_t *const new_layer_pool_max();

// layer combinators
typedef s_layer_t const *p_layer_t;
extern s_node_t *transform(context_t *, const s_layer_t *, s_node_t *);
extern s_node_t *transform_all(context_t *, p_layer_t layers[], s_node_t *);

// debug APIs
extern void s_model_info(const s_model_t *);
extern dataset_t *new_fake_dataset(const shape_t *, uint32_t);
extern void debug_tensor(const char *, const tensor_ref_t *);

// high level export APIs
typedef s_model_t *(classification_model_func_t)(context_t *, const shape_t *,
                                                 uint32_t);
typedef struct classifier_t classifier_t;
extern classifier_t *new_classifier(classification_model_func_t,
                                    const shape_t *, uint32_t);
extern void del_classifier(const classifier_t *);
extern void classifier_load(const classifier_t *, const char *,
                            const tensor_ref_t *);
extern uint32_t most_likely(const classifier_t *, const tensor_ref_t *);
extern void top_likely(const classifier_t *, const tensor_ref_t *, uint32_t,
                       int32_t *);

#ifdef __cplusplus
}
#endif
