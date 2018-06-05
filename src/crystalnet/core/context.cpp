#include <cstdarg>

#include <crystalnet/core/user_context.hpp>

context_t *new_context() { return new context_t; }

void del_context(context_t *ctx) { delete ctx; }

// shape resources

const shape_t *mk_shape(context_t *ctx, int n, ...)
{
    std::vector<uint32_t> dims;
    va_list list;
    va_start(list, n);
    for (auto i = 0; i < n; ++i) {
        uint32_t dim = va_arg(list, uint32_t);
        dims.push_back(dim);
    }
    va_end(list);
    return ctx->make_shape(dims);
}

const shape_list_t *mk_shape_list(context_t *ctx,
                                  const shape_t *const p_shapes[])
{
    std::vector<shape_t> shapes;
    for (auto p = p_shapes; *p; ++p) { shapes.push_back(**p); }
    return ctx->make_shape_list(shapes);
}

// symbol resources

s_node_t *var(context_t *ctx, const shape_t *shape)
{
    return ctx->make_placeholder(*shape);
}

s_node_t *covar(context_t *ctx, const shape_t *shape)
{
    return ctx->make_parameter(*shape);
}

s_node_t *apply(context_t *ctx, const operator_t *op, s_node_t *args[])
{
    return ctx->make_operator(*op,
                              std::vector<s_node_t *>(args, args + op->arity));
}

s_node_t *reshape(context_t *ctx, const shape_t *shape, const s_node_t *node)
{
    return ctx->wrap_node(*shape, node);
}

// layer resources

#include <crystalnet-ext.h>  // TODO: maybe move layer to core api

s_node_t *transform(context_t *ctx, const s_layer_t *l, s_node_t *x)
{
    return ctx->_layers((*l)(*ctx, x));
}

// s_model resources

s_model_t *make_s_model(context_t *ctx, s_node_t *input, s_node_t *output)
{
    return ctx->make_s_model(new s_model_t(*ctx, *input, *output));
}
