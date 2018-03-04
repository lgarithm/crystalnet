#include <cstdarg>
#include <vector>

#include <crystalnet.h>
#include <crystalnet/core/shape.hpp>

shape_t *make_shape(int n, ...)
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

void free_shape(shape_t *shape) { delete shape; }

uint32_t shape_rank(const shape_t *shape) { return shape->rank(); }

uint32_t shape_dim(const shape_t *shape) { return shape->dim(); }

shape_ctx_t *make_shape_ctx() { return new shape_ctx_t; }

void free_shape_ctx(shape_ctx_t *ctx) { delete ctx; }

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
