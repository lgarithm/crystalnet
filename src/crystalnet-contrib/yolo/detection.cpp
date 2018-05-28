#include <algorithm>
#include <cmath>

#include <crystalnet-contrib/yolo/detection.hpp>
#include <crystalnet-contrib/yolo/input.hpp>

detection_list_t get_detections(const tensor_ref_t &t,
                                const tensor_ref_t &biases)
{
    using T = float;

    const uint32_t n = 5;
    const uint32_t classes = 80;
    const uint32_t coords = 4;

    const uint32_t h = 13;
    const uint32_t w = 13;

    const auto r =
        ranked<4, T>(t.reshape(shape_t(n, coords + 1 + classes, h, w)));
    const auto e = ranked<2, T>(biases.reshape(shape_t(n, 2)));

    // [n, coords + 1 + classes, h, w]
    std::vector<std::unique_ptr<darknet::detection_t>> dets;
    for (auto l : range(n)) {
        for (auto i : range(h)) {
            for (auto j : range(w)) {
                darknet::bbox_t bbox;
                bbox.cx = (j + r.at(l, 0, i, j)) / w;
                bbox.cy = (i + r.at(l, 1, i, j)) / h;
                bbox.w = e.at(l, 0) * std::exp(r.at(l, 2, i, j)) / w;
                bbox.h = e.at(l, 1) * std::exp(r.at(l, 3, i, j)) / h;

                dets.push_back(std::make_unique<darknet::detection_t>(classes));
                const auto &d = dets[dets.size() - 1];

                d->scale = r.at(l, 4, i, j);
                d->objectness = 0;
                d->bbox = bbox;
                {
                    const auto probs = ranked<1, T>(ref(d->probs));
                    for (auto k : range(classes)) {
                        probs.at(k) = r.at(l, 5 + k, i, j);
                    }
                }
            }
        }
    }
    return dets;
}

template <typename T> struct interval {
    T lo;
    T hi;

    interval(T lo, T hi) : lo(lo), hi(hi) {}

    T clip(T x) const
    {
        if (x < lo) { return lo; }
        if (x > hi) { return hi; }
        return x;
    }
};

darknet::clip rasterize(const darknet::bbox_t &b, uint32_t h, uint32_t w)
{
    const interval<int> X(0, w);
    const interval<int> Y(0, h);

    const int left = X.clip(w * (b.cx - b.w * .5));
    int right = X.clip(w * (b.cx + b.w * .5));
    const int top = Y.clip(h * (b.cy - b.h * .5));
    int bottom = Y.clip(h * (b.cy + b.h * .5));

    if (right <= left) { right = left + 1; }
    if (bottom <= top) { bottom = top + 1; }

    darknet::clip c;
    c.left = left;
    c.right = right;
    c.top = top;
    c.bottom = bottom;
    return c;
}

void set_pixel(const ranked_tensor_ref_t<uint8_t, 3> &image,  //
               uint32_t i, uint32_t j, uint8_t g)
{
    image.at(i, j, 0) = g;
    image.at(i, j, 1) = g;
    image.at(i, j, 2) = g;
}

void draw_clip(const ranked_tensor_ref_t<uint8_t, 3> &ry,
               const darknet::clip &c)
{
    // logf("[%d, %d], [%d, %d]", c.left, c.right, c.top, c.bottom);
    const uint8_t g = 0;
    for (uint32_t i = c.left; i < c.right; ++i) {
        set_pixel(ry, c.top, i, g);
        set_pixel(ry, c.bottom - 1, i, g);
    }
    for (uint32_t i = c.top; i < c.bottom; ++i) {
        set_pixel(ry, i, c.left, g);
        set_pixel(ry, i, c.right - 1, g);
    }
}
