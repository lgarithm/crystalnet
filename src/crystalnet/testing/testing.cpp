#include <crystalnet-internal.h>
#include <crystalnet/model/model.hpp>
#include <crystalnet/model/parameter.hpp>

parameter_ctx_t *new_parameter_ctx() { return new parameter_ctx_t; }

void del_parameter_ctx(const parameter_ctx_t *ctx) { delete ctx; }

void del_model(const model_t *model) { delete model; }
