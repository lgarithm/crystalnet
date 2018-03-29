#include <cstdint>
#include <cstdio>

#include <crystalnet/core/tracer.hpp>

struct xterm_t {
    xterm_t(uint8_t b, uint8_t f) { printf("\e[%u;%um", b, f); }
    ~xterm_t() { printf("\e[m"); }
};

#define WITH_XTERM(b, f, e)                                                    \
    {                                                                          \
        xterm_t _(b, f);                                                       \
        e;                                                                     \
    }

void tracer_ctx_t::indent()
{
    for (int i = 1; i < depth; ++i) {
        printf("    ");
    }
}

tracer_ctx_t default_tracer_ctx;

tracer_t::tracer_t(const std::string &name, tracer_ctx_t &ctx)
    : name(name), t0(std::chrono::system_clock::now()), ctx(ctx)
{
    ctx.in();
    ctx.indent();
    WITH_XTERM(1, 35, printf("{ // [%s]", name.c_str()));
    putchar('\n');
}

tracer_t::~tracer_t()
{
    const auto now = std::chrono::system_clock::now();
    const std::chrono::duration<double> d = now - t0;
    char buffer[128];
    sprintf(buffer, "[%s] took %fs", name.c_str(), d.count());
    ctx.indent();
    WITH_XTERM(1, 32, printf("} // %s", buffer));
    putchar('\n');
    ctx.out();
}
