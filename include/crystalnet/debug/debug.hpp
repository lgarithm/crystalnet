#pragma once
#include <string>

#include <crystalnet-internal.h>
#include <crystalnet/core/context.hpp>
#include <stdtracer>

#define p_str(x) std::to_string(x).c_str()

void debug(const model_ctx_t &);
void debug(const model_t &);
void debug(const s_model_t &);
void show_layers(const s_model_t &);
void show_layers(const model_t &, const s_model_t &);

template <typename T, typename F>
void debug(const named_context_t<T> &ctx, const F &show_value)
{
    int width = 0;
    for (const auto &[name, _] : ctx.items) {
        if (name.size() > width) { width = name.size(); }
    }
    for (const auto &[name, value] : ctx.items) {
        logf("%-*s : %s", width, name.c_str(), show_value(value).c_str());
    }
}
