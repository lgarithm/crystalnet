#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dtypes_t dtypes_t;

struct dtypes_t {
    const uint8_t i8;
    const uint8_t u8;
    const uint8_t i16;
    const uint8_t i32;
    const uint8_t f32;
    const uint8_t f64;
};

// TODO: define dtypes const struct
// extern dtypes_t dtypes;

#define DTYPE_FLOAT 0x0D

const char *version();

typedef struct shape_t shape_t;
typedef struct shape_list_t shape_list_t;
typedef struct tensor_t tensor_t;
typedef struct model_t model_t;
typedef struct model_ctx_t model_ctx_t;
typedef struct node_t node_t;

// TODO: make it possible to add user defined operators
typedef struct forward_ctx_t forward_ctx_t;
typedef struct backward_ctx_t backward_ctx_t;
typedef struct operator_t operator_t;

typedef shape_t *(shape_func_t)(const shape_list_t *);
typedef void(forward_func_t)(forward_ctx_t *);
typedef void(backward_func_t)(backward_ctx_t *);

// shape_t *new_shape(uint8_t); // make_shape should be used instead
// void init_shape(shape_t *, uint32_t *);
shape_t *make_shape(int, ...);
void free_shape(shape_t *);

uint32_t shape_dim(const shape_t *);
uint32_t shape_rank(const shape_t *);

tensor_t *new_tensor(shape_t *, uint8_t);
void free_tensor(tensor_t *);
const shape_t *tensor_shape(tensor_t *);

model_ctx_t *new_model_ctx();
// void free_model_ctx(model_ctx_t *);  // managed by GC

model_t *new_model(model_ctx_t *, node_t *, node_t *);
void free_model(model_t *);

node_t *make_placeholder(model_ctx_t *, shape_t *);
node_t *make_parameter(model_ctx_t *, shape_t *);

typedef node_t *pnode_list_t[];
node_t *make_operator(model_ctx_t *, operator_t *, pnode_list_t);
node_t *wrap(model_ctx_t *, shape_t *, node_t *);

// operators
operator_t *register_op(const char *const, uint8_t, shape_func_t,
                        forward_func_t, backward_func_t);
extern operator_t *op_add;
extern operator_t *op_mul;
extern operator_t *op_softmax;
extern operator_t *op_xentropy;

// training
typedef struct dataset_t dataset_t;
typedef struct trainer_t trainer_t;

typedef struct optimizer_t optimizer_t;
extern optimizer_t *opt_sgd;
extern optimizer_t *opt_adam;

// dataset_t *new_dataset();
void free_dataset(dataset_t *);
trainer_t *new_trainer(model_t *, operator_t *, optimizer_t *);
void free_trainer(trainer_t *);
void run_trainer(trainer_t *, dataset_t *);
void test_trainer(trainer_t *, dataset_t *);

// dataset
dataset_t *load_mnist();
dataset_t *load_cifar();

tensor_t *_load_idx_file(const char *filename);

#ifdef __cplusplus
}
#endif
