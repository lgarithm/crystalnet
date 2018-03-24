#pragma once
#include <stdint.h>

#include <crystalnet.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trait_ctx_t trait_ctx_t;
typedef struct filter_trait_t filter_trait_t;
typedef struct padding_trait_t padding_trait_t;
typedef struct stride_trait_t stride_trait_t;

extern trait_ctx_t *new_trait_ctx();
extern void del_trait_ctx(trait_ctx_t *);
extern filter_trait_t *mk_filter(trait_ctx_t *, const shape_t *);
extern padding_trait_t *mk_padding(trait_ctx_t *, const shape_t *);
extern stride_trait_t *mk_stride(trait_ctx_t *, const shape_t *);

// layer APIs
typedef struct s_layer_t s_layer_t;
extern void del_s_layer(s_layer_t *);
extern s_layer_t *const new_layer_dense(uint32_t);
extern s_layer_t *const new_layer_relu();
extern s_layer_t *const new_layer_softmax();
extern s_layer_t *const new_layer_pool2d(const filter_trait_t *,
                                         const stride_trait_t *);
extern s_layer_t *const new_layer_conv2d(const filter_trait_t *,
                                         const padding_trait_t *,
                                         const stride_trait_t *);

// TODO: deprecate
extern s_layer_t *const new_layer_conv_nhwc(uint32_t, uint32_t, uint32_t);
extern s_layer_t *const new_layer_pool_max();

// layer combinators
typedef s_layer_t const *p_layer_t;
extern s_node_t *transform(s_model_ctx_t *, const s_layer_t *, s_node_t *);
extern s_node_t *transform_all(s_model_ctx_t *, p_layer_t layers[], s_node_t *);

// debug APIs
extern void s_model_info(const s_model_t *);
extern dataset_t *new_fake_dataset(const shape_t *, uint32_t);

// high level export APIs
typedef s_model_t *(classification_model_func_t)(const shape_t *, uint32_t);
typedef struct classifier_t classifier_t;
extern classifier_t *new_classifier(classification_model_func_t,
                                    const shape_t *, uint32_t);
extern void del_classifier(const classifier_t *);
extern void classifier_load(const classifier_t *, const char *,
                            const tensor_ref_t *);
extern uint32_t most_likely(const classifier_t *, const tensor_ref_t *);

#ifdef __cplusplus
}
#endif
