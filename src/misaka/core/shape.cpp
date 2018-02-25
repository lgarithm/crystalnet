#include <cstdarg>
#include <cstdio>

#include <misaka.h>
#include <misaka/core/shape.hpp>

shape_t *new_shape(uint8_t rank) { return new shape_t(rank); }

shape_t *make_shape(int n, ...)
{
    shape_t *shape = new shape_t(n);
    va_list list;
    va_start(list, n);
    for (auto i = 0; i < n; ++i) {
        uint32_t dim = va_arg(list, uint32_t);
        shape->dims[i] = dim;
    }
    va_end(list);
    return shape;
}

void free_shape(shape_t *shape) { delete shape; }

uint32_t shape_rank(const shape_t *shape) { return shape->rank(); }

uint32_t shape_dim(const shape_t *shape) { return shape->dim(); }

void init_shape(shape_t *shape, uint32_t *dims)
{
    const auto rank = shape->rank();
    for (auto i = 0; i < rank; ++i) {
        shape->dims[i] = dims[i];
    }
}
