#pragma once
#include <misaka/core/gc.hpp>
#include <misaka/core/tensor.hpp>

#include <memory>
#include <utility>

namespace
{
auto gc = GC<tensor_t>();
}

struct dataset_t {
    using item_t = std::pair<tensor_ref_t, tensor_ref_t>;

    virtual item_t next() = 0;

    // virtual item_t next_barch() {} // TODO: next_batch

    virtual bool has_next() { return false; }
    virtual ~dataset_t() {}
};

struct fake_dataset_t : dataset_t {
    uint64_t step = 0;
    fake_dataset_t() {}

    item_t next() override
    {
        DEBUG(__func__);
        auto shape = shape_t(1);
        auto t1 = gc(new tensor_t(shape));
        auto t2 = gc(new tensor_t(shape));
        return item_t(ref(*t1), ref(*t2));
    }

    bool has_next() override
    {
        DEBUG(__func__);
        step++;
        return step < 10;
    }
};

struct simple_dataset_t : dataset_t {
    using owner_t = std::unique_ptr<tensor_t>;

    int idx;
    uint32_t n;

    owner_t images;
    owner_t labels;

    simple_dataset_t(tensor_t *images, tensor_t *labels)
        : images(owner_t(images)), labels(owner_t(labels)), idx(0),
          n(images->shape.len())
    {
        DEBUG(__func__);
    }

    bool has_next() override { return idx < n; }

    item_t next() override
    {
        DEBUG(__func__);
        bool ok = has_next();
        assert(ok);
        auto i = idx++;
        return item_t((*images)[i], (*labels)[i]);
    }
};
