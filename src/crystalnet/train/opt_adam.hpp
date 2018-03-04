#pragma once
#include <cmath>
#include <memory>

#include <crystalnet/train/optimizer.hpp>

struct adam_optimizer_t : optimizer_t {
    struct ctx : optimizer_ctx_t {
        struct p_ctx {
            using T = float;
            static constexpr T default_learning_rate = T(1e-3);
            const T eps = 1e-8;
            const T alpha = default_learning_rate;
            const T beta_1 = 0.9;
            const T beta_2 = 0.999;

            T beta_1_t = 1;
            T beta_2_t = 1;

            const node_t *node;
            tensor_ref_t _value;
            tensor_ref_t _gradient;
            tensor_t _delta;
            tensor_t _moment;
            tensor_t _velocity;

            uint32_t step;

            p_ctx(const node_t *node)
                : node(node), _value(node->value()),
                  _gradient(node->gradient()), _delta(node->shape, node->dtype),
                  _moment(node->shape, node->dtype),
                  _velocity(node->shape, node->dtype), step(0)
            {
            }

            T _learn_scalar(const T &g, T &m, T &v)
            {
                m = beta_1 * m + (1 - beta_1) * g;
                v = beta_2 * v + (1 - beta_2) * g * g;
                return -alpha * m / (1 - beta_1_t) /
                       (std::sqrt(v / (1 - beta_2_t)) + eps);
            }

            void operator()()
            {
                // printf("%s %d\n", __func__, ++step);
                using ref_t = r_tensor_ref_t<T>;
                ref_t gradient(_gradient);
                ref_t delta(_delta);
                ref_t moment(_moment);
                ref_t velocity(_velocity);
                const auto n = gradient.shape.dim();
                beta_1_t *= beta_1;
                beta_2_t *= beta_2;
                for (auto i = 0; i < n; ++i) {
                    delta.data[i] = _learn_scalar(
                        gradient.data[i], moment.data[i], velocity.data[i]);
                }

                ref_t value(_value);
                for (auto i = 0; i < n; ++i) {
                    value.data[i] += delta.data[i];
                }
            }
        };

        // model_t *model;
        std::vector<std::unique_ptr<p_ctx>> p_ctxs;

        ctx(model_t *model)
        {
            for (auto node : model->ctx->params.items) {
                p_ctxs.push_back(std::make_unique<p_ctx>(node));
            }
        }

        void operator()() override
        {
            for (auto &p : p_ctxs) {
                (*p)();
            }
        }
    };

    optimizer_ctx_t *optimize(model_t *model) override
    {
        return new ctx(model);
    }
};
