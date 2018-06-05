#include <stdexcept>
#include <string>

#include <crystalnet-internal.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/model/model.hpp>
#include <crystalnet/symbol/model.hpp>
#include <crystalnet/symbol/node.hpp>

model_t *realize(parameter_ctx_t *p_ctx, const s_model_t *m,
                 uint32_t batch_size)
{
    model_option_t opt(m->input.name, batch_size);
    model_ctx_t *ctx = gc(new model_ctx_t(*p_ctx));
    const auto output = m->output.realize(*ctx, opt);
    const auto places = ctx->places.items;
    if (places.size() != 1) {
        // TODO: support any number of placeholders
        throw std::logic_error(
            "exactly one placeholder should be specified, got " +
            std::to_string(places.size()));
    }
    return new model_t(*ctx, *places[0], *output);
}
