#include <memory>

#include <crystalnet-ext.h>
#include <crystalnet-internal.h>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/core/tracer.hpp>
#include <crystalnet/debug/debug.hpp>
#include <crystalnet/ops/argmax.hpp>
#include <crystalnet/ops/batch.hpp>
#include <crystalnet/symbol/model.hpp>
#include <crystalnet/utility/range.hpp>

struct classifier_t {
    const shape_t image_shape;
    const uint32_t class_number;

    parameter_ctx_t p_ctx;

    using s_model_owner_t = std::unique_ptr<s_model_t>;
    const s_model_owner_t s_model;

    using mode_owner_t = std::unique_ptr<model_t>;
    const mode_owner_t model;

    classifier_t(classification_model_func_t func, const shape_t &image_shape,
                 const uint32_t class_number)
        : image_shape(image_shape), class_number(class_number),
          s_model(func(&image_shape, class_number)),
          model(realize(&p_ctx, s_model.get(), 1)) // TODO: support batch
    {
    }

    void load(const std::string &name, const tensor_ref_t &r) const
    {
        p_ctx.load(name, r);
    }

    std::vector<int32_t> top_likely(const tensor_ref_t &input, uint32_t k) const
    {
        TRACE(__func__);
        using T = float;
        k = std::min(k, class_number);
        model->input.bind(embed(input));
        TRACE_IT(model->output.forward());
        TRACE_IT(debug(*model));
        const auto output = ranked<2, T>(model->output.value());
        tensor_t _indexes(shape_t(k), idx_type<int32_t>::type);
        const auto indexes = ranked<1, int32_t>(ref(_indexes));
        top_indexes(output[0], indexes);
        std::vector<int32_t> result;
        for (auto i : range(k)) {
            const auto idx = indexes.data[i];
            printf("[d] %d %f\n", idx, output[0].data[idx]);
            result.push_back(idx);
        }
        return result;
    }

    uint32_t most_likely(const tensor_ref_t &input) const
    {
        TRACE(__func__);
        model->input.bind(embed(input));
        model->output.forward();
        using T = float;
        r_tensor_ref_t<T> output(model->output.value());
        return argmax(r_tensor_ref_t<T>(output));
    }
};

classifier_t *new_classifier(classification_model_func_t model,
                             const shape_t *shape, uint32_t class_number)
{
    return new classifier_t(model, *shape, class_number);
}

void del_classifier(const classifier_t *classifier) { delete classifier; }

void classifier_load(const classifier_t *classifier, const char *name,
                     const tensor_ref_t *r)
{
    classifier->load(name, *r);
}

uint32_t most_likely(const classifier_t *c, const tensor_ref_t *input)
{
    return c->most_likely(*input);
}

void top_likely(const classifier_t *c, const tensor_ref_t *input, uint32_t k,
                int32_t *result)
{
    const auto r = c->top_likely(*input, k);
    for (auto i : range(r.size())) {
        result[i] = r[i];
    }
}
