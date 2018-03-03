#pragma once
#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>

#include <crystalnet/core/tensor.hpp>

struct dataset_t {
    using item_t = std::pair<tensor_ref_t, tensor_ref_t>;

    virtual item_t next() = 0;
    // virtual item_t next_barch() {} // TODO: next_batch
    virtual void reset() = 0;
    virtual bool has_next() const = 0;
    virtual const shape_t *image_shape() const = 0;
    virtual const shape_t *label_shape() const = 0;
    virtual ~dataset_t() {}
};

struct range_t {
    using item_t = dataset_t::item_t;
    dataset_t &ds;

    explicit range_t(dataset_t &ds) : ds(ds) {}

    struct iter_t {
        dataset_t *ds;
        std::unique_ptr<item_t> next;
        explicit iter_t(dataset_t *ds) : ds(ds) { this->operator++(); }
        bool operator!=(const iter_t &it) const { return ds != it.ds; }
        void operator++()
        {
            if (ds && ds->has_next()) {
                auto item = new item_t(ds->next());
                next.reset(item);
            } else {
                ds = nullptr;
            }
        }
        item_t operator*() { return item_t(*next); }
    };

    iter_t begin()
    {
        if (ds.has_next()) {
            return iter_t(&ds);
        }
        return end();
    }

    iter_t end() { return iter_t(nullptr); }
};

inline range_t range(dataset_t &ds)
{
    ds.reset();
    return range_t(ds);
}

struct simple_dataset_t : dataset_t {
    using owner_t = std::unique_ptr<tensor_t>;

    int idx; // TODO: move idx to iterator
    const uint32_t n;
    const shape_t _image_shape;
    const shape_t _label_shape;
    owner_t images;
    owner_t labels;

    simple_dataset_t(tensor_t *images, tensor_t *labels)
        : _image_shape(images->shape.sub()), _label_shape(labels->shape.sub()),
          images(owner_t(images)), labels(owner_t(labels)), idx(0),
          n(images->shape.len())
    {
    }

    bool has_next() const override { return idx < n; }
    void reset() override { idx = 0; }
    item_t next() override
    {
        assert(has_next());
        auto i = idx++;
        return item_t(ref(*images)[i], ref(*labels)[i]);
    }
    const shape_t *image_shape() const override { return &_image_shape; }
    const shape_t *label_shape() const override { return &_label_shape; }
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
            assert(false);
        }
    }
    return distro_;
}

template <typename T> void normalize(const r_tensor_ref_t<T> &r, T b)
{
    std::transform(r.data, r.data + r.shape.dim(), r.data,
                   [&](T x) { return x / b; });
}
