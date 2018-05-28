// https://github.com/pjreddie/darknet.git

#include <float.h>
#include <math.h>
#include <memory.h>

#include <crystalnet-contrib/darknet/darknet.h>
#include <crystalnet-contrib/yolo/logistic.h>

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i * stride] > largest) largest = input[i * stride];
    }
    for (i = 0; i < n; ++i) {
        float e = exp(input[i * stride] / temp - largest / temp);
        sum += e;
        output[i * stride] = e;
    }
    for (i = 0; i < n; ++i) { output[i * stride] /= sum; }
}

void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups,
                 int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for (b = 0; b < batch; ++b) {
        for (g = 0; g < groups; ++g) {
            softmax(input + b * batch_offset + g * group_offset, n, temp,
                    stride, output + b * batch_offset + g * group_offset);
        }
    }
}

typedef struct {
    int background;
    int softmax;
    int softmax_tree;

    int batch;
    int w;
    int h;
    int n;

    float *output;
    int outputs;
    int inputs;

    int coords;
    int classes;
} layer;

void activate_array(float *x, const int n)
{
    for (int i = 0; i < n; ++i) { x[i] = logistic_activate(x[i]); }
}

int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w * l.h);
    int loc = location % (l.w * l.h);
    return batch * l.outputs + n * l.w * l.h * (l.coords + l.classes + 1) +
           entry * l.w * l.h + loc;
}

void forward_region_layer(float *input, int batch, float *output, int outputs,
                          int _n, int w, int h)
{
    layer l;
    {
        l.background = 0;
        l.softmax = 1;
        l.softmax_tree = 0;

        l.batch = batch;
        l.w = w;
        l.h = h;
        l.n = _n;
        l.output = output;
        l.outputs = outputs;
        l.inputs = outputs;

        l.coords = 4;
        l.classes = 80;
    }

    int i, j, b, t, n;
    memcpy(l.output, input, l.outputs * l.batch * sizeof(float));

    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n * l.w * l.h, 0);
            activate_array(l.output + index, 2 * l.w * l.h);

            index = entry_index(l, b, n * l.w * l.h, l.coords);
            if (!l.background) activate_array(l.output + index, l.w * l.h);

            index = entry_index(l, b, n * l.w * l.h, l.coords + 1);
            if (!l.softmax && !l.softmax_tree)
                activate_array(l.output + index, l.classes * l.w * l.h);
        }
    }

    if (l.softmax_tree) {
        // int i;
        // int count = l.coords + 1;
        // for (i = 0; i < l.softmax_tree->groups; ++i) {
        //     int group_size = l.softmax_tree->group_size[i];
        //     softmax_cpu(net.input + count, group_size, l.batch, l.inputs,
        //                 l.n * l.w * l.h, 1, l.n * l.w * l.h, l.temperature,
        //                 l.output + count);
        //     count += group_size;
        // }
    } else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_cpu(input + index, l.classes + l.background, l.batch * l.n,
                    l.inputs / l.n, l.w * l.h, 1, l.w * l.h, 1,
                    l.output + index);
    }
}
