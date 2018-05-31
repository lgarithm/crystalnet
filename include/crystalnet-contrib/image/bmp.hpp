#pragma once
#include <crystalnet/core/tensor.hpp>

extern tensor_t *read_bmp_file(const char * /* filename */);
extern void write_bmp_file(const char * /* filename */,
                           const ranked_tensor_ref_t<uint8_t, 3> & /* image */);
