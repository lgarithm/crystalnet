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
