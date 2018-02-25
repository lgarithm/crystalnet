#include <misaka/core/debug.hpp>
#include <misaka/core/gc.hpp>
#include <misaka/model/model.hpp>

static auto gc = GC<model_ctx_t>();

model_ctx_t *new_model_ctx() { return gc(new model_ctx_t); }

void free_model_ctx(model_ctx_t *model) { delete model; }

node_t *make_placeholder(model_ctx_t *model, shape_t *shape)
{
    return model->make_placeholder(*shape);
}

node_t *make_parameter(model_ctx_t *model, shape_t *shape)
{
    return model->make_parameter(*shape);
}

node_t *make_operator(model_ctx_t *model, operator_t *op, node_t *nodes[])
{
    return model->make_operator(*op, nodes, op->name.c_str());
}

node_t *wrap(model_ctx_t *model, shape_t *shape, node_t *node)
{
    return model->wrap(*shape, *node);
}

model_t *new_model(model_ctx_t *ctx, node_t *input, node_t *output)
{
    DEBUG(__func__);
    return new model_t(ctx, input, output);
}

void free_model(model_t *model) { delete model; }
