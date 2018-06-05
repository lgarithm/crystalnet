#include <crystalnet.h>
#include <crystalnet/core/context.hpp>
#include <crystalnet/core/layer.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/symbol/model.hpp>

struct context_t : generic_context_t<shape_t>,
                   generic_context_t<shape_list_t>,
                   s_model_ctx_t,
                   named_context_t<s_model_t> {

    context_t() : named_context_t<s_model_t>("s_model") {}

    const shape_t *make_shape(const std::vector<uint32_t> &dims)
    {
        return generic_context_t<shape_t>::gc(new shape_t(dims));
    }

    const shape_list_t *make_shape_list(const std::vector<shape_t> &shapes)
    {
        return generic_context_t<shape_list_t>::gc(new shape_list_t(shapes));
    }

    s_model_t *make_s_model(s_model_t *model)
    {
        // TODO: support multiple s_model per context_t
        return named_context_t<s_model_t>::own(model, "s_model");
    }
};
