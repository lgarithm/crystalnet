// The IDX file format: http://yann.lecun.com/exdb/mnist/
#pragma once
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <misaka/core/error.hpp>
#include <misaka/core/tensor.hpp>
#include <misaka/data/dataset.hpp>

struct idx_meta {
    uint8_t type;
    uint8_t rank;

    explicit idx_meta(uint8_t type, uint8_t rank) : type(type), rank(rank) {}

    explicit idx_meta(uint32_t magic)
        : type((magic >> 8) & 0xff), rank(magic & 0xff)
    {
    }
};

uint32_t magic_word(uint8_t type, uint8_t rank)
{
    return (uint32_t)type * 256 + rank;
}

struct idx_file_header {
    uint32_t magic;
    uint32_t *sizes;
};

void reverse_byte_order(uint32_t &x)
{
    x = (x << 24) | ((x << 8) & 0xff0000) | ((x >> 8) & 0xff00) | (x >> 24);
}

tensor_t *load_idx_file(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        EXIT("File Not Found");
    }
    uint32_t magic;
    fread(&magic, 4, 1, fp);
    reverse_byte_order(magic);
    idx_meta meta(magic);
    shape_t shape(meta.rank);
    for (auto i = 0; i < meta.rank; ++i) {
        uint32_t dim;
        fread(&dim, 4, 1, fp);
        reverse_byte_order(dim);
        shape.dims[i] = dim;
    }
    auto tensor = new tensor_t(shape, meta.type);
    fread(tensor->data, shape.dim(), dtype_size(meta.type), fp);
    fclose(fp);
    return tensor;
}
