#include <crystalnet-ext.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/ext/traits.hpp>

struct trait_ctx_t {
    GC<filter_trait_t> gc_filter;
    GC<padding_trait_t> gc_padding;
    GC<stride_trait_t> gc_stride;

    filter_trait_t *make_filter(const shape_t &shape)
    {
        return gc_filter(new filter_trait_t(shape));
    }

    padding_trait_t *make_padding(const shape_t &shape)
    {
        return gc_padding(new padding_trait_t(shape));
    }

    stride_trait_t *make_stride(const shape_t &shape)
    {
        return gc_stride(new stride_trait_t(shape));
    }
};

trait_ctx_t *new_trait_ctx() { return new trait_ctx_t; }

void del_trait_ctx(trait_ctx_t *ctx) { delete ctx; }

filter_trait_t *mk_filter(trait_ctx_t *ctx, const shape_t *shape)
{
    return ctx->make_filter(*shape);
}

padding_trait_t *mk_padding(trait_ctx_t *ctx, const shape_t *shape)
{
    return ctx->make_padding(*shape);
}

stride_trait_t *mk_stride(trait_ctx_t *ctx, const shape_t *shape)
{
    return ctx->make_stride(*shape);
}
