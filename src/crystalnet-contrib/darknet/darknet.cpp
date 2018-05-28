// https://github.com/pjreddie/darknet.git

#include <cfloat>
#include <string>

#include <crystalnet-contrib/darknet/darknet.h>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/core/tracer.hpp>

std::string summary(float *x, int n)
{
    tensor_ref_t _r(shape_t(n), dtypes.f32, x);
    r_tensor_ref_t<float> r(_r);
    return summary(r);
}

void reorg_cpu(float *x, int w, int h, int c, int batch, int stride,
               int forward, float *out)
{
    int b, i, j, k;
    int out_c = c / (stride * stride);

    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h; ++j) {
                for (i = 0; i < w; ++i) {
                    int in_index = i + w * (j + h * (k + c * b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i * stride + offset % stride;
                    int h2 = j * stride + offset / stride;
                    int out_index =
                        w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));
                    if (forward)
                        out[out_index] = x[in_index];
                    else
                        out[in_index] = x[out_index];
                }
            }
        }
    }
}

float im2col_get_pixel(float *im, int height, int width, int channels, int row,
                       int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float *data_im, int channels, int height, int width, int ksize,
                int stride, int pad, float *data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] =
                    im2col_get_pixel(data_im, height, width, channels, im_row,
                                     im_col, c_im, pad);
            }
        }
    }
}
