#include <cstdarg>
#include <vector>

#include <crystalnet-internal.h>
#include <crystalnet/core/context.hpp>
#include <crystalnet/core/shape.hpp>

const shape_t *new_shape(int n, ...)
{
    std::vector<uint32_t> dims;
    va_list list;
    va_start(list, n);
    for (auto i = 0; i < n; ++i) {
        uint32_t dim = va_arg(list, uint32_t);
        dims.push_back(dim);
    }
    va_end(list);
    return new shape_t(dims);
}

void del_shape(const shape_t *shape) { delete shape; }

uint32_t shape_rank(const shape_t *shape) { return shape->rank(); }

uint32_t shape_dim(const shape_t *shape) { return shape->dim(); }

struct shape_ctx_t : generic_context_t<shape_t>,
                     generic_context_t<shape_list_t> {

    const shape_t *make_shape(const std::vector<uint32_t> &dims)
    {
        return generic_context_t<shape_t>::gc(new shape_t(dims));
    }

    const shape_list_t *make_shape_list(const std::vector<shape_t> &shapes)
    {
        return generic_context_t<shape_list_t>::gc(new shape_list_t(shapes));
    }
};

shape_ctx_t *new_shape_ctx() { return new shape_ctx_t; }

void del_shape_ctx(shape_ctx_t *ctx) { delete ctx; }

const shape_t *mk_shape(shape_ctx_t *ctx, int n, ...)
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

const shape_list_t *mk_shape_list(shape_ctx_t *ctx,
                                  const shape_t *const p_shapes[])
{
    std::vector<shape_t> shapes;
    for (auto p = p_shapes; *p; ++p) { shapes.push_back(**p); }
    return ctx->make_shape_list(shapes);
}
