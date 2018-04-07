#include <crystalnet.h>

#include <crystalnet/core/idx.hpp>
#include <crystalnet/core/tensor.hpp>

const dtypes_t dtypes = {
    idx_type<uint8_t>::type, //
    idx_type<int8_t>::type,  //
    idx_type<int16_t>::type, //
    idx_type<int32_t>::type, //
    idx_type<float>::type,   //
    idx_type<double>::type,  //
};

void _reverse_byte_order(uint32_t &x)
{
    x = (x << 24) | ((x << 8) & 0xff0000) | ((x >> 8) & 0xff00) | (x >> 24);
}

_tensor_meta_t _read_idx_header(FILE *fp)
{
    uint8_t magic[4];
    check(std::fread(&magic, 4, 1, fp) == 1); // [0, 0, dtype, rank]
    const uint8_t rank = magic[3];
    std::vector<uint32_t> dims(rank);
    for (auto i = 0; i < rank; ++i) {
        check(std::fread(&dims[i], 4, 1, fp) == 1);
        _reverse_byte_order(dims[i]);
    }
    return _tensor_meta_t(magic[2], shape_t(dims));
}

tensor_t *_load_idx_file(const char *filename)
{
    FILE *fp = std::fopen(filename, "r");
    check(fp != nullptr);
    const auto meta = _read_idx_header(fp);
    const auto tensor = new tensor_t(meta.shape, meta.dtype);
    const auto count = meta.shape.dim();
    check(std::fread(tensor->data, dtype_size(meta.dtype), count, fp) == count);
    std::fclose(fp);
    return tensor;
}

void _idx_file_info(const char *filename)
{
    FILE *fp = std::fopen(filename, "r");
    check(fp != nullptr);
    const auto meta = _read_idx_header(fp);
    printf("%s%s\n", dtype_name(meta.dtype),
           std::to_string(meta.shape).c_str());
    std::fclose(fp);
}
