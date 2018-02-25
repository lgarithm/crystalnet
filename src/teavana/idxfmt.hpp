// The IDX file format: http://yann.lecun.com/exdb/mnist/
#pragma once

#include <array>   // for array
#include <cassert> // for assert
#include <cstddef> // for size_t
#include <cstdint> // for uint8_t, uint32_t, int16_t, int32_t, int8_t
#include <cstdio>  // for FILE, fclose, fopen, ftell, fread, fseek, etc
#include <string>  // for string

#include "teavana/core/shape.hpp" // for shape_t
#include "teavana/core/tensor.hpp" // for tensor, tensor_ref (ptr only), data_size

namespace tea
{
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

uint8_t element_size(const idx_meta &mt)
{
    switch (mt.type) {
    case 0x08:
        return 1;
    case 0x09:
        return 1;
    case 0x0B:
        return 2;
    case 0x0C:
        return 4;
    case 0x0D:
        return 4;
    case 0x0E:
        return 8;
    default:
        assert(false);
        return 0;
    }
}

template <uint8_t r> struct idx_file_header {
    static const uint8_t rank = r;

    uint32_t magic;
    uint32_t sizes[r];
};

template <uint8_t r> size_t size(idx_file_header<r> &h)
{
    size_t s = 1;
    for (uint8_t i = 0; i < r; ++i) {
        s *= h.sizes[i];
    }
    return s;
}

void reverse_byte_order(uint32_t &x)
{
    x = (x << 24) | ((x << 8) & 0xff0000) | ((x >> 8) & 0xff00) | (x >> 24);
}

template <uint8_t r> void reverse_byte_order(idx_file_header<r> &hdr)
{
    reverse_byte_order(hdr.magic);
    for (uint8_t i = 0; i < r; ++i) {
        reverse_byte_order(hdr.sizes[i]);
    }
}

template <uint8_t r> void read(idx_file_header<r> &hdr, FILE *fp)
{
    constexpr size_t size = sizeof(hdr);
    static_assert(size == 4 * (r + 1), "invalid size");
    auto _ = fread(&hdr, size, 1, fp);
    reverse_byte_order(hdr);
}

template <typename R, uint8_t r> tensor<R, r> load_idx_file(FILE *fp)
{
    fseek(fp, 0, SEEK_END);
    const size_t file_size = ftell(fp);
    rewind(fp);

    idx_file_header<r> hdr;
    read(hdr, fp);
    const size_t data_size = size(hdr) * element_size(idx_meta(hdr.magic));
    assert(ftell(fp) + data_size == file_size);

    ::std::array<size_t, r> dims;
    for (uint8_t i = 0; i < r; ++i) {
        dims[i] = hdr.sizes[i];
    }
    tensor<R, r> t((shape_t<r>(dims)));
    auto _ = fread(t.data, data_size, 1, fp);
    fclose(fp);
    return t;
}

template <typename R, uint8_t r>
tensor<R, r> load_idx_file_from(const ::std::string &name)
{
    FILE *fp = fopen(name.c_str(), "r");
    assert(fp != nullptr);
    return load_idx_file<R, r>(fp);
}

template <typename> struct idx_type;

template <> struct idx_type<uint8_t> {
    static constexpr uint8_t type = 0x08;
};

template <> struct idx_type<int8_t> {
    static constexpr uint8_t type = 0x09;
};

template <> struct idx_type<int16_t> {
    static constexpr uint8_t type = 0x0B;
};

template <> struct idx_type<int32_t> {
    static constexpr uint8_t type = 0x0C;
};

template <> struct idx_type<float> {
    static constexpr uint8_t type = 0x0D;
};

template <> struct idx_type<double> {
    static constexpr uint8_t type = 0x0E;
};

template <typename R, uint8_t r>
void save_tensor(const tensor_ref<R, r> &t, FILE *fp)
{
    idx_file_header<r> hdr;
    hdr.magic = magic_word(idx_type<R>::type, r);
    for (uint8_t i = 0; i < r; ++i) {
        hdr.sizes[i] = t.shape.dims[i];
    }
    reverse_byte_order(hdr);

    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(t.data, data_size(t), 1, fp);
}

template <typename R, uint8_t rank>
void save_tensor_to(const tensor_ref<R, rank> &t, const ::std::string &name)
{
    FILE *fp = fopen(name.c_str(), "w");
    assert(fp != nullptr);
    save_tensor(t, fp);
    fclose(fp);
}

template <typename R, uint8_t r>
void load_tensor(const tensor_ref<R, r> &t, FILE *fp)
{
    idx_file_header<r> hdr;
    read(hdr, fp);

    idx_meta meta(hdr.magic);
    assert(meta.rank == r);
    assert(meta.type == idx_type<R>::type);

    for (uint8_t i = 0; i < r; ++i) {
        assert(t.shape.dims[i] == hdr.sizes[i]);
    }
    auto _ = fread(t.data, dim(t.shape) * sizeof(R), 1, fp);
}
}
