#pragma once
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

#include <crystalnet/core/tensor.hpp>

template <typename T> T adj_diff_sum(const T *x, uint32_t n)
{
    T y = 0;
    for (uint32_t i = 1; i < n; ++i) { y += std::fabs(x[i] - x[i - 1]); }
    return y;
}

struct fletcher32_t {
    // https://en.wikipedia.org/wiki/Fletcher%27s_checksum#Optimizations
    uint32_t fletcher32(const uint16_t *data, size_t len) const
    {
        uint32_t c0 = 0;
        uint32_t c1 = 0;

        for (c0 = c1 = 0; len >= 360; len -= 360) {
            for (int i = 0; i < 360; ++i) {
                c0 = c0 + *data++;
                c1 = c1 + c0;
            }
            c0 = c0 % 65535;
            c1 = c1 % 65535;
        }
        for (int i = 0; i < len; ++i) {
            c0 = c0 + *data++;
            c1 = c1 + c0;
        }
        c0 = c0 % 65535;
        c1 = c1 % 65535;
        return (c1 << 16 | c0);
    }

    template <typename T> uint32_t operator()(const T *data, uint32_t n) const
    {
        static_assert(sizeof(T) % 2 == 0);
        return fletcher32((const uint16_t *)data, n * sizeof(T) / 2);
    }
};

template <typename T> struct tensor_summary_t {
    const uint32_t dim;
    const T min;
    const T mean;
    const T max;
    const T std;
    const T adj_diff_sum;
    const uint32_t check_sum;

    explicit tensor_summary_t(const r_tensor_ref_t<T> &r)
        : dim(r.shape.dim()), min(r.min()), mean(r.mean()), max(r.max()),
          std(r.std()), adj_diff_sum(::adj_diff_sum(r.data, r.shape.dim())),
          check_sum(fletcher32_t()(r.data, r.shape.dim()))
    {
    }
};

struct hex_t {
    uint32_t x;
};

inline std::ostream &operator<<(std::ostream &os, const hex_t &h)
{
    std::stringstream ss;
    ss << "0x" << std::setfill('0') << std::setw(8) << std::hex << h.x;
    return os << ss.str();
}

namespace std
{
inline string to_string(const tensor_ref_t &t)
{
    return dtype_name(t.dtype) + to_string(t.shape);
}

inline string to_string(const tensor_t &t) { return to_string(ref(t)); }

template <typename T> inline string to_string(const tensor_summary_t<T> &s)
{
    stringstream ss;
    ss << std::fixed                          //
       << "check_sum=" << hex_t{s.check_sum}  //
       << ", [" << s.min << ", " << s.max << "]"
       << ", mean=" << s.mean                  //
       << ", std=" << s.std                    //
       << ", adj_diff_sum=" << s.adj_diff_sum  //
       << ", dim=" << s.dim                    //
        ;
    return ss.str();
}
}  // namespace std

template <typename T> auto summary(const r_tensor_ref_t<T> &r)
{
    return std::to_string(tensor_summary_t<T>(r));
}

template <typename T, uint8_t r>
auto summary(const ranked_tensor_ref_t<T, r> &t)
{
    std::vector<uint32_t> dims(r);
    std::copy(t.shape.dims.begin(), t.shape.dims.end(), dims.begin());
    return summary(r_tensor_ref_t<T>(shape_t(dims), t.data));
}
