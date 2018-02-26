#include <cstdint>

#include <misaka.h>
#include <misaka/core/idx.hpp>

const dtypes_t dtypes = {
    idx_type<uint8_t>::type, //
    idx_type<int8_t>::type,  //
    idx_type<int16_t>::type, //
    idx_type<int32_t>::type, //
    idx_type<float>::type,   //
    idx_type<double>::type,  //
};
