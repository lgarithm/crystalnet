#pragma once
#include <sstream>
#include <vector>

#include <crystalnet-ext.h>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/core/tracer.hpp>
#include <crystalnet/utility/range.hpp>

namespace darknet
{
template <typename T> struct bbox {
    T cx;
    T cy;
    T h;
    T w;
};

using bbox_t = bbox<float>;

struct clip {
    uint32_t left;
    uint32_t right;
    uint32_t top;
    uint32_t bottom;
};

struct detection_t {
    using T = float;

    bbox_t bbox;

    tensor_t probs;

    float objectness = 0;
    float scale = 0;

    explicit detection_t(int classes)
        : probs(shape_t(classes), idx_type<T>::type)
    {
    }
};
}  // namespace darknet

using detection_list_t = std::vector<std::unique_ptr<darknet::detection_t>>;
detection_list_t get_detections(const tensor_ref_t & /* results */,
                                const tensor_ref_t & /* anchor boxes */);

extern darknet::clip rasterize(const darknet::bbox_t &b, uint32_t h,
                               uint32_t w);
extern void draw_clip(const ranked_tensor_ref_t<uint8_t, 3> &,
                      const darknet::clip &);

namespace std
{
inline string to_string(const darknet::bbox_t &bb)
{
    stringstream ss;
    ss << "bbox<"
       << "cx=" << bb.cx << ","
       << "cy=" << bb.cy << ","
       << "w=" << bb.w << ","
       << "h=" << bb.h << ">";
    return ss.str();
}
}  // namespace std
