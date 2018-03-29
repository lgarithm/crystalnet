#include "imagenet.h"

#include <iostream>
#include <numeric>

#include <crystalnet-ext.h>
#include <opencv2/opencv.hpp>

void info(const cv::Mat &img, const std::string &name)
{
    std::cout << "info: " << name << std::endl;
    std::cout << "type: " << img.type() << std::endl; //
    std::cout << "channels: " << img.channels() << std::endl;
    std::cout << "depth: " << img.depth() << std::endl;
    std::cout << "elemSize: " << img.elemSize() << std::endl;
    std::cout << "elemSize1: " << img.elemSize1() << std::endl;
    const auto s = img.size();
    std::cout << "size: " << s.height << " X " << s.width << std::endl;
}

struct square_normalizer {
    const int r;

    explicit square_normalizer(const int r) : r(r) {}

    cv::Mat operator()(const cv::Mat &img) const
    {
        const auto size = img.size();

        const auto d = std::min(size.width, size.height);
        const int g = std::gcd(d, r);
        const int p = d / g;
        const int q = r / g;

        const cv::Size new_size(size.width * q / p, size.height * q / p);
        cv::Mat dst(new_size, CV_8UC(3));
        cv::resize(img, dst, dst.size(), 0, 0);

        cv::Mat out(cv::Size(r, r), CV_8UC(3));
        cv::getRectSubPix(
            dst, out.size(),
            cv::Point2f(new_size.width * .5, new_size.height * .5), out);

        return out;
    }
};

cv::Mat square_normalize(const cv::Mat &img, int r)
{
    const square_normalizer normalize(r);
    return normalize(img);
}

std::size_t to_hwc(const cv::Mat &img, const tensor_ref_t *tensor)
{
    const auto len = img.dataend - img.data;
    assert(tensor_dtype(tensor) == dtypes.u8);
    memcpy(tensor_data_ptr(tensor), img.data, len);
    return len;
}
