#include <crystalnet-contrib/image/bmp.hpp>

#include <cstdint>
#include <cstdio>
#include <memory>

// https://en.wikipedia.org/wiki/BMP_file_format#Pixel_storage
uint32_t row_size(uint32_t width, uint8_t bitspp)
{
    return ((width * bitspp + 31) / 32) * 4;
}

struct bmp_header_t {
    uint32_t file_size;      // file size in bytes
    uint16_t _creater1 = 0;  // 0
    uint16_t _creater2 = 0;  // 0
    uint32_t offset = 54;    // offset to image data
};

struct bmp_info_t {
    uint32_t header_size = 40;   // info size in bytes
    uint32_t width;              // width in pixel
    uint32_t height;             // height in pixel
    uint16_t nplanes = 1;        // number of color planes
    uint16_t bitspp = 24;        // bits per pixel
    uint32_t compress_type = 0;  // compress type
    uint32_t image_size;         // image size in bytes
    uint32_t hres = 0;           // pixels per meter
    uint32_t vres = 0;           // pixels per meter
    uint32_t ncolors = 0;        // number of colors
    uint32_t nimpcolors = 0;     // important colors
};

void copy_slice(uint8_t *dest, const uint8_t *src, int n, int stride)
{
    for (int i = 0; i < n; ++i) { dest[i * stride] = src[i * stride]; }
}

tensor_t *read_bmp_file(const char *filename)
{
    FILE *fp = std::fopen(filename, "r");
    if (fp == nullptr) {
        throw std::runtime_error(std::string("can't open ") + filename);
    }
    char magic[2];
    if (std::fread(magic, 2, 1, fp) != 1) {
        throw std::runtime_error("bad bmp file: bad magic");
    }
    bmp_header_t header;
    if (std::fread(&header, sizeof(bmp_header_t), 1, fp) != 1) {
        throw std::runtime_error("bad bmp file: header too short");
    }
    bmp_info_t info;
    if (std::fread(&info, sizeof(bmp_info_t), 1, fp) != 1) {
        throw std::runtime_error("bad bmp file: info too short");
    }
    if (info.bitspp != 24) { throw std::runtime_error("unsupported bmp type"); }
    const uint32_t rs = row_size(info.width, 24);
    const shape_t shape(info.height, info.width, 3);
    const int n = shape.dim();
    uint8_t row[rs];
    uint8_t pixels[n];
    uint8_t *p = pixels + n;
    for (int i = 0; i < info.height; ++i) {
        if (std::fread(row, rs, 1, fp) != 1) {
            throw std::runtime_error("bad bmp file: data too short");
        }
        p -= info.width * 3;
        copy_slice(p + 0, row + 2, info.width, 3);
        copy_slice(p + 1, row + 1, info.width, 3);
        copy_slice(p + 2, row + 0, info.width, 3);
    }
    // TODO: check next of fp is eof
    // if (!std::feof(fp)) {
    //     throw std::runtime_error("bad bmp file: data too long");
    // }
    std::fclose(fp);
    using T = float;
    tensor_t *_t = new tensor_t(shape, idx_type<T>::type);
    const r_tensor_ref_t<T> r(*_t);
    for (int i = 0; i < n; ++i) { r.data[i] = pixels[i] / 255.0; }
    return _t;
}

void write_bmp_file(const char *filename,
                    const ranked_tensor_ref_t<uint8_t, 3> &image)
{
    const auto [h, w, c] = image.shape.dims;
    if (c != 3) { throw std::invalid_argument("channel must be 3"); }
    FILE *fp = std::fopen(filename, "wb");
    if (fp == nullptr) {
        throw std::runtime_error(std::string("can't open ") + filename);
    }
    static const char magic[2] = {'B', 'M'};
    if (std::fwrite(magic, 2, 1, fp) != 1) {
        throw std::runtime_error("failed to write");
    }
    const uint32_t rs = row_size(w, 24);
    bmp_header_t header;
    header.file_size = rs * h + 54;
    if (std::fwrite(&header, sizeof(bmp_header_t), 1, fp) != 1) {
        throw std::runtime_error("failed to write");
    }
    bmp_info_t info;
    info.width = w;
    info.height = h;
    info.image_size = rs * h;
    if (std::fwrite(&info, sizeof(bmp_info_t), 1, fp) != 1) {
        throw std::runtime_error("failed to write");
    }
    uint8_t row[rs];
    std::memset(row, 0, sizeof(row));
    for (int i = 0; i < h; ++i) {
        const auto p = image[h - i - 1];
        copy_slice(row + 0, p.data + 2, w, 3);
        copy_slice(row + 1, p.data + 1, w, 3);
        copy_slice(row + 2, p.data + 0, w, 3);
        if (std::fwrite(row, rs, 1, fp) != 1) {
            throw std::runtime_error("failed to write");
        }
    }
    std::fclose(fp);
}
