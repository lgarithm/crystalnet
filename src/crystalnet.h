#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const char *version();

typedef struct dtypes_t dtypes_t;

struct dtypes_t {
    const uint8_t i8;
    const uint8_t u8;
    const uint8_t i16;
    const uint8_t i32;
    const uint8_t f32;
    const uint8_t f64;
};
extern const dtypes_t dtypes;

typedef struct shape_t shape_t;
typedef struct shape_ctx_t shape_ctx_t;
typedef struct tensor_t tensor_t;
typedef struct tensor_ref_t tensor_ref_t;
typedef struct operator_t operator_t;

extern const shape_t *new_shape(int, ...);
extern void del_shape(const shape_t *);
extern uint32_t shape_dim(const shape_t *);
extern uint32_t shape_rank(const shape_t *);
extern shape_ctx_t *new_shape_ctx();
extern void del_shape_ctx(shape_ctx_t *);
extern const shape_t *mk_shape(shape_ctx_t *, int, ...);

extern tensor_t *new_tensor(const shape_t *, uint8_t);
extern void del_tensor(const tensor_t *);
extern const tensor_ref_t *tensor_ref(const tensor_t *);
extern void *tensor_data_ptr(const tensor_ref_t *);
extern const shape_t *tensor_shape(const tensor_ref_t *);
extern const uint8_t tensor_dtype(const tensor_ref_t *);

// symbolic APIs
typedef struct s_node_t s_node_t;
typedef s_node_t *symbol;
typedef struct s_model_t s_model_t;
typedef struct s_model_ctx_t s_model_ctx_t;
extern s_model_ctx_t *make_s_model_ctx();
extern s_model_t *new_s_model(s_model_ctx_t *, s_node_t *, s_node_t *);
extern void del_s_model(s_model_t *);
extern s_node_t *var(s_model_ctx_t *, const shape_t *);
extern s_node_t *covar(s_model_ctx_t *, const shape_t *);
extern s_node_t *reshape(s_model_ctx_t *, const shape_t *, const s_node_t *);
extern s_node_t *apply(s_model_ctx_t *, const operator_t *, s_node_t *args[]);

// high level APIs
typedef struct dataset_t dataset_t;
typedef struct optimizer_t optimizer_t;
typedef struct s_trainer_t s_trainer_t;

// dataset_t *new_dataset();
extern void del_dataset(dataset_t *);
extern const shape_t *ds_image_shape(dataset_t *);
extern const shape_t *ds_label_shape(dataset_t *);

extern s_trainer_t *new_s_trainer(const s_model_t *, const operator_t *,
                                  const optimizer_t *);
extern void del_s_trainer(s_trainer_t *);
extern void s_trainer_run(s_trainer_t *, dataset_t *);
extern void s_rtainer_test(s_trainer_t *, dataset_t *);

// dataset
extern dataset_t *load_mnist(const char *const); // train | t10k
extern dataset_t *load_cifar();

// unstable APIs
extern tensor_t *_load_idx_file(const char *);
extern void _idx_file_info(const char *);
extern void s_experiment(s_trainer_t *, dataset_t *, dataset_t *, uint32_t);

// operators
extern const operator_t *op_add;
extern const operator_t *op_mul;
extern const operator_t *op_relu;
extern const operator_t *op_softmax;
extern const operator_t *op_xentropy;
// unstable operators:
extern const operator_t *op_conv_nhwc;
extern const operator_t *op_pool2d_c_max;
extern const operator_t *make_op_pool2d(uint32_t, uint32_t, uint32_t, uint32_t);
extern const operator_t *make_op_conv2d(uint32_t, uint32_t, uint32_t, uint32_t);

// optimizers
extern const optimizer_t *opt_sgd;
extern const optimizer_t *opt_adam;

#ifdef __cplusplus
}
#endif
