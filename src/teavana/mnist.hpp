// Reader for http://yann.lecun.com/exdb/mnist/
#pragma once

#include <cstdint> // for uint8_t
#include <memory>  // for unique_ptr
#include <string>  // for string, operator+

#include "teavana/bmp.hpp"         // for write_bmp_file, bmp_head
#include "teavana/core/shape.hpp"  // for shape, operator*, shape_t
#include "teavana/core/tensor.hpp" // for tensor, transform_tensor
#include "teavana/idxfmt.hpp"      // for load_idx_file_from
#include "teavana/utils.hpp"       // for dirac_distribution
#include "teavana/zip.hpp"         // for zip

namespace tea
{
namespace mnist
{

constexpr auto image_shape = shape(28, 28);
constexpr auto distro_shape = shape(10);

struct data_source {
    const ::std::string data_file;
    const ::std::string label_file;

    data_source(const ::std::string &data_file, const ::std::string &label_file)
        : data_file(data_file), label_file(label_file)
    {
    }

    static data_source named(const ::std::string &name)
    {
        return data_source(name + "-images-idx3-ubyte",
                           name + "-labels-idx1-ubyte");
    }
};

template <typename R>
tensor<R, 2> transform_labels(const tensor<uint8_t, 1> &labels)
{
    tensor<R, 2> distros(labels.shape * shape(10));
    for (const auto &p : zip(labels, distros)) {
        dirac_distribution(scalar(p.first), p.second);
    }
    return distros;
}

template <typename R> struct data_set {
    // const shape_t<2> image_shape = shape(28, 28);
    const tensor<R, 3> images;
    const tensor<R, 2> distros;

    data_set(const data_source &ds)
        : images(transform_tensor<uint8_t, R>(
              [](uint8_t p) { return p / 255.0; },
              load_idx_file_from<uint8_t, 3>(ds.data_file))),
          distros(transform_labels<R>(
              load_idx_file_from<uint8_t, 1>(ds.label_file)))
    {
    }
};

struct image {
    ::std::unique_ptr<uint8_t[]> data;
    image() : data(new uint8_t[28 * 28]) {}
};

void write_image_file(const image &img, const ::std::string &filename)
{
    unsigned char buffer[28 * 28 * 3];
    int idx = 0;
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            uint8_t pix = img.data[(28 - 1 - i) * 28 + j];
            buffer[idx++] = pix;
            buffer[idx++] = pix;
            buffer[idx++] = pix;
        }
    }
    write_bmp_file(bmp_head(28, 28), buffer, filename.c_str());
}
} // namespace mnist
}
