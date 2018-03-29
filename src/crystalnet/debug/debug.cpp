#include <crystalnet-ext.h>
#include <crystalnet-internal.h>
#include <crystalnet/debug/debug.hpp>
#include <crystalnet/model/model.hpp>

void debug(const std::string &name, const tensor_ref_t &t)
{
    using T = float;
    r_tensor_ref_t<T> r(t);
    printf("%-32s: ", name.c_str());
    print(r);
}

void debug_tensor(const char *name, const tensor_ref_t *t)
{
    printf("%-32s: %s %s\n", name, std::to_string(*t).c_str(),
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
    printf("parameters:\n");
    for (auto p : ctx.params.items) {
        debug(*p);
    }
}

void print_opertors(const model_ctx_t &ctx)
{
    printf("operators:\n");
    for (auto o : ctx.ops.items) {
        debug(*o);
    }
}

void debug(const model_ctx_t &ctx)
{
    printf("debug:\n");
    print_parameters(ctx);
    print_opertors(ctx);
}

void debug(const model_t &m) { debug(m.ctx); }
