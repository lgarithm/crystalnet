#include <crystalnet-internal.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/model/model.hpp>
#include <crystalnet/symbol/model.hpp>
#include <crystalnet/symbol/node.hpp>

static GC<s_model_ctx_t> gc;

s_model_ctx_t *make_s_model_ctx() { return gc(new s_model_ctx_t); }

s_node_t *var(s_model_ctx_t *ctx, const shape_t *shape)
{
    return ctx->make_placeholder(*shape);
}

s_node_t *covar(s_model_ctx_t *ctx, const shape_t *shape)
{
    return ctx->make_parameter(*shape);
}

s_node_t *apply(s_model_ctx_t *ctx, const operator_t *op, s_node_t *args[])
{
    std::vector<s_node_t *> nodes;
    for (auto i = 0; i < op->arity; ++i) {
        nodes.push_back(args[i]);
    }
    return ctx->make_operator(*op, nodes);
}

s_node_t *reshape(s_model_ctx_t *ctx, const shape_t *shape,
                  const s_node_t *node)
{
    return ctx->wrap_node(*shape, node);
}

s_model_t *new_s_model(s_model_ctx_t *ctx, s_node_t *input, s_node_t *output)
{
    return new s_model_t(ctx, input, output);
}

void del_s_model(s_model_t *model) { delete model; }

model_t *realize(parameter_ctx_t *p_ctx, const s_model_t *m,
                 uint32_t batch_size)
{
    static GC<model_ctx_t> gc;
    printf("[D] realising s_model_t\n");
    model_option_t opt(m->input, batch_size);
    model_ctx_t *ctx = gc(new model_ctx_t(p_ctx));
    auto output = m->output->realize(*ctx, opt);
    auto places = ctx->places.items;
    if (places.size() != 1) {
        // TODO: support any number of placeholders
        printf("exact one placeholder must be specified!\n");
        check(false);
    }
    auto inputs = places[0];
    printf("[D] realized s_model_t\n");
    return new model_t(ctx, inputs, output);
}
