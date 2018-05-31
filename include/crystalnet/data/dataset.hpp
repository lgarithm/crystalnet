#pragma once
#include <algorithm>
#include <memory>
#include <utility>

#include <crystalnet/core/error.hpp>
#include <crystalnet/core/tensor.hpp>

struct dataset_t {
    using item_t = std::pair<tensor_ref_t, tensor_ref_t>;
    virtual const shape_t *image_shape() const = 0;
    virtual const shape_t *label_shape() const = 0;
    virtual const uint32_t len() const = 0;
    virtual item_t operator[](uint32_t) const = 0;
    virtual item_t slice(uint32_t, uint32_t) const = 0;
    virtual ~dataset_t() {}

    struct iter_t {
        const dataset_t &ds;
        uint32_t pos;
        explicit iter_t(const dataset_t &ds, uint32_t pos) : ds(ds), pos(pos) {}
        bool operator!=(const iter_t &it) const { return pos != it.pos; }
        void operator++() { ++pos; }
        item_t operator*() const { return ds[pos]; }
    };

    iter_t begin() const { return iter_t(*this, 0); }

    iter_t end() const { return iter_t(*this, len()); }
};

struct batched_dataset_t {
    using item_t = std::pair<tensor_ref_t, tensor_ref_t>;
    const dataset_t &ds;
    const uint32_t batch_size;
    const uint32_t _total_size;
    const uint32_t end_pos;

    static uint32_t align(uint32_t n, uint32_t m) { return n / m * m; }

    batched_dataset_t(const dataset_t &ds, uint32_t batch_size)
        : ds(ds), batch_size(batch_size), _total_size(ds.len()),
          end_pos(align(_total_size, batch_size))
    {
        // TODO: make trainer and model support dynamic batch size.
        fprintf(stderr, "%u data points dropped\n", _total_size - end_pos);
    }

    struct iter_t {
        const dataset_t &ds;
        const uint32_t batch_size;
        uint32_t pos;
        explicit iter_t(const dataset_t &ds, uint32_t batch_size, uint32_t pos)
            : ds(ds), batch_size(batch_size), pos(pos)
        {
        }
        bool operator!=(const iter_t &it) const { return pos != it.pos; }
        void operator++() { pos += batch_size; }
        item_t operator*() const { return ds.slice(pos, pos + batch_size); }
    };

    iter_t begin() const { return iter_t(ds, batch_size, 0); }
    iter_t end() const { return iter_t(ds, batch_size, end_pos); }
};

inline batched_dataset_t batch(const dataset_t &ds, uint32_t batch_size)
{
    return batched_dataset_t(ds, batch_size);
}

struct simple_dataset_t : dataset_t {
    using owner_t = std::unique_ptr<tensor_t>;

    const uint32_t n;
    const shape_t _image_shape;
    const shape_t _label_shape;
    owner_t images;
    owner_t labels;

    simple_dataset_t(tensor_t *images, tensor_t *labels)
        : _image_shape(images->shape.sub()), _label_shape(labels->shape.sub()),
          images(owner_t(images)), labels(owner_t(labels)),
          n(images->shape.len())
    {
    }
    const shape_t *image_shape() const override { return &_image_shape; }
    const shape_t *label_shape() const override { return &_label_shape; }
    const uint32_t len() const override { return n; }
    item_t operator[](uint32_t i) const override
    {
        return item_t(ref(*images)[i], ref(*labels)[i]);
    }
    item_t slice(uint32_t i, uint32_t j) const override
    {
        return item_t(ref(*images).slice(i, j), ref(*labels).slice(i, j));
    }
};

inline std::string data_dir()
{
    return std::string(getenv("HOME")) + "/var/data";
}

template <typename T> tensor_t *make_onehot(const tensor_t &tensor, uint32_t k)
{
    auto dims = std::vector<uint32_t>(tensor.shape.dims);
    dims.push_back(k);
    auto distro_ = new tensor_t(shape_t(dims), idx_type<T>::type);
    r_tensor_ref_t<T> distro(*distro_);
    r_tensor_ref_t<uint8_t> r(tensor); // TODO: support other uint types
    auto n = tensor.shape.dim();
    for (auto i = 0; i < n; ++i) {
        auto off = r.data[i];
        if (0 <= off && off < k) {
            distro.data[i * k + off] = 1;
        } else {
            // TODO: print a warning msg
            check(false);
        }
    }
    return distro_;
}

template <typename T> void normalize(const r_tensor_ref_t<T> &r, T b)
{
    std::transform(r.data, r.data + r.shape.dim(), r.data,
                   [&](T x) { return x / b; });
}
