#include <misaka/core/debug.hpp>

#include <cstdio>

#include <misaka/core/shape.hpp>

void log_func_call(const char *name) { printf("%s called\n", name); }

void log_tensor_usage(const shape_t *shape, uint8_t dtype_size)
{
    // printf("new tensor: rank %d, dim: %d, dtype size: %d\n", shape->rank(),
    //        shape->dim(), dtype_size);
}

void log_node_usage(const shape_t *shape, const std::string &name)
{
    printf("new node: %s, rank %d, dim: %d, shape: %s\n", name.c_str(),
           shape->rank(), shape->dim(), std::to_string(*shape).c_str());
}
