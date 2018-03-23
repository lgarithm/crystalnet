#pragma once
#include <utility>

#include <crystalnet-internal.h>
#include <crystalnet/data/dataset.hpp>
#include <crystalnet/model/model.hpp>
#include <crystalnet/ops/argmax.hpp>
#include <crystalnet/train/optimizer.hpp>
#include <crystalnet/utility/range.hpp>

struct s_trainer_t {
    static constexpr uint32_t default_batch_size = 9999;

    parameter_ctx_t p_ctx;
    const s_model_t *const model;
    operator_t *const loss_func;
    optimizer_t *const optimizer;

    static node_t *make_label(model_t *model)
    {
        return model->ctx->make_placeholder(model->output->shape);
    }

    static node_t *make_loss(model_t *model, node_t *label,
                             operator_t *loss_func)
    {
        node_t *args[] = {label, model->output};
        return model->ctx->make_operator(*loss_func, args,
                                         loss_func->name.c_str());
    }

    s_trainer_t(const s_model_t *model, operator_t *loss_func,
                optimizer_t *optimizer)
        : model(model), loss_func(loss_func), optimizer(optimizer)
    {
    }

    void run(dataset_t &ds, dataset_t *test_ds = nullptr,
             uint32_t batch_size = default_batch_size)
    {
        auto m = realize(&p_ctx, model, batch_size);
        std::unique_ptr<model_t> __m(m);
        auto label = make_label(m);
        auto loss = make_loss(m, label, loss_func);
        auto optimize = optimizer->optimize(m);
        std::unique_ptr<optimizer_ctx_t> __o(optimize);

        printf("[D] training, batch size: %u\n", batch_size);
        uint32_t step = 0;
        for (auto[images, label_s] : batch(ds, batch_size)) {
            ++step;
            printf("[D] begin step %u\n", step);
            m->input->bind(images);
            label->bind(label_s);
            loss->forward();
            r_tensor_ref_t<float>(loss->gradient()).fill_uni();
            loss->backward();
            (*optimize)();
            m->ctx->debug();
            printf("train step: %u\n", step);
            if (test_ds) {
                const auto[yes, tot] = test(*test_ds);
                printf("test acc: %g\n", yes / (float)tot);
            }
        }
    }

    std::pair<uint32_t, uint32_t> test(dataset_t &ds)
    {
        const auto batch_size = 1000;
        uint32_t no = 0;
        uint32_t yes = 0;
        auto m = realize(&p_ctx, model, batch_size);
        std::unique_ptr<model_t> __m(m);
        uint32_t step = 0;
        for (auto[images, label_s] : batch(ds, batch_size)) {
            ++step;
            m->input->bind(images);
            m->output->forward();
            using T = float;
            for (auto i : range(batch_size)) {
                auto p = argmax(r_tensor_ref_t<T>(label_s[i]));
                auto q = argmax(r_tensor_ref_t<T>(m->output->value()[i]));
                p == q ? ++yes : ++no;
            }
            printf("test step: %u, %u/%u\n", step, yes, yes + no);
        }
        return std::make_pair(yes, yes + no);
    }
};
