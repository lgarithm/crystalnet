#pragma once
#include <cstdint>

inline uint8_t dtype_size(uint8_t dtype)
{
    switch (dtype) {
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

// TODO: use template variable if possible

template <typename> struct idx_type;

template <> struct idx_type<uint8_t> {
    static constexpr uint8_t type = 0x08;
    static constexpr const char *const name = "u8";
};

template <> struct idx_type<int8_t> {
    static constexpr uint8_t type = 0x09;
    static constexpr const char *const name = "i8";
};

template <> struct idx_type<int16_t> {
    static constexpr uint8_t type = 0x0B;
    static constexpr const char *const name = "i16";
};

template <> struct idx_type<int32_t> {
    static constexpr uint8_t type = 0x0C;
    static constexpr const char *const name = "i32";
};

template <> struct idx_type<float> {
    static constexpr uint8_t type = 0x0D;
    static constexpr const char *const name = "f32";
};

template <> struct idx_type<double> {
    static constexpr uint8_t type = 0x0E;
    static constexpr const char *const name = "f64";
};

inline const char *dtype_name(uint8_t dtype)
{
    switch (dtype) {
    case 0x08:
        return idx_type<uint8_t>::name;
    case 0x09:
        return idx_type<int8_t>::name;
    case 0x0B:
        return idx_type<int16_t>::name;
    case 0x0C:
        return idx_type<int32_t>::name;
    case 0x0D:
        return idx_type<float>::name;
    case 0x0E:
        return idx_type<double>::name;
    default:
        static constexpr const char *const _name = "invalid";
        return _name;
    }
}
