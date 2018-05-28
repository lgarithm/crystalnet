#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <experimental/filesystem>
#include <getopt.h>

namespace fs = std::experimental::filesystem;

#include <crystalnet.h>

extern "C" {
#include <darknet.h>
extern char *get_layer_string(LAYER_TYPE a);
}

#include "options.hpp"

std::string layer_name(LAYER_TYPE lt) { return get_layer_string(lt); }

struct saver_t {
    const fs::path path;
    explicit saver_t(const fs::path &path) : path(path) {}

    void operator()(const std::string &name, const tensor_ref_t *r) const
    {
        const auto filename = path / (name + ".idx");
        save_tensor(filename.c_str(), r);
        printf("[i] %s saved to %s\n", name.c_str(), filename.c_str());
    }
};

struct converter_t {
    const std::string name_prefix;
    const saver_t save;

    converter_t(const fs::path &path, const std::string &name_prefix)
        : save(path), name_prefix(name_prefix)
    {
    }

    template <typename T>  // assert(T == float);
    void _save_tensor(const std::string &name, const shape_t *shape,
                      void *data) const
    {
        // idx_type<T>::type == dtypes.f32
        const uint32_t dtype = dtypes.f32;
        const tensor_t *tensor = new_tensor(shape, dtype);
        std::memcpy(tensor_data_ptr(tensor_ref(tensor)), data,
                    shape_dim(shape) * sizeof(T));
        save(name, tensor_ref(tensor));
        del_tensor(tensor);
    }

    void save_conv_layer(const std::string &name, const layer &l) const
    {
        using T = float;
        {
            const shape_t *bias_shape = new_shape(1, l.n);
            _save_tensor<T>(name + "_b", bias_shape, l.biases);
            if (l.batch_normalize && (!l.dontloadscales)) {
                _save_tensor<T>(name + "_scales", bias_shape, l.scales);
                _save_tensor<T>(name + "_rolling_means", bias_shape,
                                l.rolling_mean);
                _save_tensor<T>(name + "_rolling_variances", bias_shape,
                                l.rolling_variance);
            }
            del_shape(bias_shape);
        }
        {
            const shape_t *shape = new_shape(4, l.n, l.c, l.size, l.size);
            if (l.nweights != shape_dim(shape)) {
                fprintf(stderr, "invalid layer\n");
                exit(1);
            }
            _save_tensor<T>(name + "_W", shape, l.weights);
            del_shape(shape);
        }
    }

    void save_region_layer(const std::string &name, const layer &l) const
    {
        using T = float;
        const shape_t *shape = new_shape(2, l.n, 2);
        _save_tensor<T>(name + "_anchors", shape, l.biases);
        del_shape(shape);
    }

    std::string lpad(const std::string &s, int width, char ch) const
    {
        return std::string(std::max<int>(width - s.size(), 0), ch) + s;
    }

    void save_parameters(const network *net) const
    {
        printf("saving weights ...\n");
        for (int i = 0; i < net->n; ++i) {
            const auto l = net->layers[i];
            const std::string prefix =
                name_prefix + "_" + lpad(std::to_string(i), 2, '0');
            switch (l.type) {
            case CONVOLUTIONAL:
                save_conv_layer(prefix, l);
                continue;
            case REGION:
                save_region_layer(prefix, l);
                continue;
            case MAXPOOL:
            case REORG:
            case ROUTE:
                printf("[d] %s layer has no weights\n",
                       layer_name(l.type).c_str());
                continue;
            default:
                // TODO: save other layers
                printf("TODO: layer %-4d: %s\n", i, layer_name(l.type).c_str());
            }
        }
        printf("done.\n");
    }

    void operator()(const fs::path &model_cfg,
                    const fs::path &weights_file) const
    {
        network *net = load_network((char *)model_cfg.c_str(),
                                    (char *)weights_file.c_str(), 0);
        printf("%d layers\n", net->n);
        save_parameters(net);
    }
};

int main(int argc, char *argv[])
{
    const auto opts = parse_flags(argc, argv);
    converter_t convert(opts.model_dir, "yolov2");
    convert(opts.model_cfg, opts.weights_file);
    return 0;
}
