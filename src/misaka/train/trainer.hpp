#pragma once
#include <misaka.h>
#include <misaka/data/dataset.hpp>
#include <misaka/linag/base.hpp>
#include <misaka/model/model.hpp>

struct trainer_t {
    model_t *model;
    node_t *label;
    node_t *loss;
    optimizer_t *optimizer;

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

    trainer_t(model_t *model, operator_t *loss_func, optimizer_t *optimizer)
        : model(model), label(make_label(model)),
          loss(make_loss(model, label, loss_func)), optimizer(optimizer)
    {
        DEBUG(__func__);
        printf("%lu parameters\n", model->ctx->params.size());
    }

    void debug(const char *name)
    {
        printf("%s\n", name);
        model->ctx->debug();
        {
            auto l = loss->value();
            r_tensor_ref_t<float> r(l);
            printf("loss: ");
            print(r);
        }
    }

    void run(dataset_t &ds)
    {
        DEBUG(__func__);
        int step = 0;
        for (auto[image, label_] : range(ds)) {
            ++step;
            model->input->bind(image);
            label->bind(label_);
            loss->forward();
            {
                r_tensor_ref_t<float> r(loss->gradient());
                auto n = r.shape.dim();
                for (auto i = 0; i < n; ++i) {
                    r.data[i] = 1;
                }
            }
            loss->backward();

            for (auto p : model->ctx->params) {
                p->learn(); // TODO: sync parameter with other agents
            }

            constexpr auto freq = 1000;
            if (step % freq == 0) {
                printf("step: %d\n", step);
                debug("steps ...");
            }
        }
    }

    void test(dataset_t &ds)
    {
        DEBUG(__func__);
        uint32_t step = 0;
        uint32_t yes = 0;
        for (auto[image, label_] : range(ds)) {
            ++step;
            model->input->bind(image);
            label->bind(label_);
            loss->forward();
            using T = float;
            auto p = argmax<T>(as_vector_ref<T>(label_));
            auto q = argmax<T>(as_vector_ref<T>(model->output->value()));
            yes += p == q;
            if (step % 10000 == 0) {
                printf("step: %d\n", step);
            }
        }
        printf("step : %u, acc : %f\n", step, (float)yes / step);
    }

    static constexpr uint32_t default_batch_size = 500;

    void run_batch(dataset_t &ds)
    {
        DEBUG(__func__);
        // TODO
    }
};
