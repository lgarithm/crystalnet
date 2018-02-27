#pragma once
#include <misaka/train/optimizer.hpp>

struct sgd_optimizer_t : optimizer_t {
    struct ctx : optimizer_ctx_t {
        model_t *model;
        ctx(model_t *model) : model(model) {}
        void operator()() override
        {
            for (auto p : model->ctx->params) {
                p->learn(); // TODO: sync parameter with other agents
            }
        }
    };

    optimizer_ctx_t *optimize(model_t *model) override
    {
        return new ctx(model);
    }
};
