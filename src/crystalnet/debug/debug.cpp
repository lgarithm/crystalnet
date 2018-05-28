#include <crystalnet-ext.h>
#include <crystalnet-internal.h>
#include <crystalnet/debug/debug.hpp>
#include <crystalnet/model/model.hpp>
#include <crystalnet/symbol/model.hpp>

void debug(const std::string &name, const tensor_ref_t &t)
{
    using T = float;
    r_tensor_ref_t<T> r(t);
    logf("%-32s: %s", name.c_str(), summary(r).c_str());
}

void debug_tensor(const char *name, const tensor_ref_t *t)
{
    logf("%-32s: %s %s\n", name, std::to_string(*t).c_str(),
         // TODO: support all types
         summary(r_tensor_ref_t<float>(*t)).c_str());
}

void debug(const node_t &n)
{
    debug(n.name, n.value());
    // constexpr const char *partial = "\xe2\x88\x82"; // TODO: use wchar
    constexpr const char *partial = "grad";
    debug(partial + (" " + n.name), n.gradient());
}

void print_parameters(const model_ctx_t &ctx)
{
    TRACE(__func__);
    for (auto p : ctx.params.items) { debug(*p); }
}

void print_opertors(const model_ctx_t &ctx)
{
    TRACE(__func__);
    for (auto o : ctx.ops.items) { debug(*o); }
}

void debug(const model_ctx_t &ctx)
{
    TRACE(__func__);
    print_parameters(ctx);
    print_opertors(ctx);
}

void debug(const model_t &m) { debug(m.ctx); }

void debug(const s_model_t &m)
{
    TRACE("debug s_model_t");
    {
        TRACE("debug s_model_t::placeholders");
        for (const auto it : m.ctx.places.items) {
            logf("%-32s %s", it->name.c_str(), p_str(it->shape));
        }
    }
    {
        TRACE("debug s_model_t::parameters");
        for (const auto it : m.ctx.params.items) {
            logf("%-32s %s", it->name.c_str(), p_str(it->shape));
        }
    }
    {
        TRACE("debug s_model_t::operators");
        for (const auto it : m.ctx.ops.items) {
            logf("%-32s %s", it->name.c_str(), p_str(it->shape));
        }
    }
}

void show_layers(const s_model_ctx_t &ctx)
{
    uint32_t idx = 0;
    for (const auto l : ctx._layers.items) {
        logf("layer %6d %-32s: %s", idx++, l->name.c_str(), p_str(l->shape));
    }
}

void show_layers(const s_model_t &m) { show_layers(m.ctx); }

void show_layers(const model_ctx_t &ctx, const s_model_ctx_t &s_ctx)
{
    uint32_t idx = 0;
    logf("%d layers", s_ctx._layers.items.size());
    for (const auto l : s_ctx._layers.items) {
        const auto node = ctx.index.at(l->name);
        using T = float;
        const auto r = r_tensor_ref_t<T>(node->value());
        const auto brief = summary(r);
        // logf("layer %6d  %-16s  %-32s    %s", idx++, l->name.c_str(),
        //      std::to_string(l->shape).c_str(), brief.c_str());
        logf("layer %-3d %s", idx++, summary(r).c_str());
    }
}

void show_layers(const model_t &m, const s_model_t &s)
{
    show_layers(m.ctx, s.ctx);
}
