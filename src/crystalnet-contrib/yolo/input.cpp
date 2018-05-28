#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

#include <crystalnet-contrib/image/bmp.hpp>
#include <crystalnet-contrib/yolo/input.hpp>
#include <crystalnet-contrib/yolo/yolo.h>
#include <crystalnet/core/shape.hpp>   // TODO: don't include private headers
#include <crystalnet/core/tensor.hpp>  // TODO: don't include private headers

namespace fs = std::experimental::filesystem;

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

const square_normalizer normalize(yolov2_input_size);

const tensor_t *image_to_chw(const cv::Mat image)
{
    const auto size = image.size();
    const uint32_t c = 3;
    const uint32_t h = size.height;
    const uint32_t w = size.width;
    const shape_t shape(c, h, w);

    using T = float;
    const tensor_t *t = new tensor_t(shape, idx_type<T>::type);
    const auto r = ranked<3, T>(ref(*t));
    for (uint32_t i = 0; i < h; ++i) {
        for (uint32_t j = 0; j < w; ++j) {
            const auto pix = image.at<cv::Vec3b>(i, j);
            r.at(0, i, j) = pix[2] / 255.0;
            r.at(1, i, j) = pix[1] / 255.0;
            r.at(2, i, j) = pix[0] / 255.0;
        }
    }
    return t;
}

const tensor_t *load_test_image(const char *filename)
{
    const auto img = cv::imread(filename);
    const auto input_image = normalize(img);
    return image_to_chw(input_image);
}

std::string extension(const fs::path &p)
{
    // TODO: use p.extension() when it's available
    const auto s = p.string();
    return s.substr(s.rfind('.'));
}

const tensor_t *load_resized_test_image(const char *filename)
{
    const auto ext = extension(fs::path(filename));
    if (ext == ".bmp") {
        const auto t = std::unique_ptr<tensor_t>(read_bmp_file(filename));
        return hwc_to_chw<float>(ref(*t));
    } else if (ext == ".idx") {
        return _load_idx_file(filename);
    } else {
        throw std::invalid_argument("unknown file format");
    }
}

bool file_exists(const fs::path &filename)
{
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp == NULL) { return false; }
    fclose(fp);
    return true;
}

void load_parameters(const model_t *model, const fs::path &model_dir)
{
    for (const auto [name, p] : model->ctx.p_ctx.items) {
        const auto filename = model_dir / (name + ".idx");
        if (!file_exists(filename)) {
            throw std::runtime_error(filename.string() + " not exist");
        }
        const auto w = _load_idx_file(filename.c_str());
        model->ctx.p_ctx.load(name, ref(*w));
    }
}

std::vector<std::string> load_name_list(const fs::path &filename)
{
    logf("%s", filename.c_str());
    std::vector<std::string> names;
    std::string line;
    std::ifstream in(filename);
    while (std::getline(in, line)) { names.push_back(line); }
    logf("got %d names", names.size());
    return names;
}
