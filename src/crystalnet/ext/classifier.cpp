#include <memory>

#include <crystalnet-ext.h>
#include <crystalnet-internal.h>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/ops/argmax.hpp>
#include <crystalnet/ops/batch.hpp>
#include <crystalnet/symbol/model.hpp>

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
        // TODO: load weight to p_ctx
        p_ctx.debug();
    }

    void load(const std::string &name, const tensor_ref_t &r) const
    {
        p_ctx.load(name, r);
    }

    uint32_t most_likely(const tensor_ref_t &input) const
    {
        model->input->bind(embed(input));
        model->output->forward();
        using T = float;
        r_tensor_ref_t<T> output(model->output->value());
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
