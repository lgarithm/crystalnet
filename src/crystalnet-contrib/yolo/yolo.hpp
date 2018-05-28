#pragma once
#include <string>
#include <vector>

#include <crystalnet-ext.h>
#include <crystalnet/symbol/model.hpp>

inline int get_layer_number(const s_model_ctx_t &ctx)
{
    return ctx._layers.items.size();
}

inline std::string name_prefix(const s_model_ctx_t &ctx)
{
    const int layer_number = get_layer_number(ctx);
    char name_prefix[32];
    sprintf(name_prefix, "yolov2_%02d", layer_number);
    return name_prefix;
}
