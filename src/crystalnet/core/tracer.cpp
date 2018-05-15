#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

#include <unistd.h>

#include <crystalnet/core/tracer.hpp>

struct xterm_t {
    const bool is_tty;
    xterm_t(uint8_t b, uint8_t f) : is_tty(isatty(fileno(stdout)))
    {
        if (is_tty) { printf("\e[%u;%um", b, f); }
    }

    ~xterm_t()
    {
        if (is_tty) { printf("\e[m"); }
    }
};

#define WITH_XTERM(b, f, e)                                                    \
    {                                                                          \
        xterm_t _(b, f);                                                       \
        e;                                                                     \
    }

tracer_ctx_t::~tracer_ctx_t()
{
    using item_t = std::tuple<duration_t, uint32_t, std::string>;
    std::vector<item_t> list;
    for (const auto [name, duration] : total_durations) {
        list.push_back(item_t(duration, call_times[name], name));
    }
    std::sort(list.rbegin(), list.rend());

    const std::string hr(80, '-');
    printf("\tsummary of %s::%s\n", "tracer_ctx_t", name.c_str());
    printf("%s\n", hr.c_str());
    printf("%8s    %16s    %s\n", "count", "total duration", "name");
    printf("%s\n", hr.c_str());
    for (const auto &[duration, count, name] : list) {
        printf("%8d    %16fs    %s\n",  //
               count, duration.count(), name.c_str());
    }
}

void tracer_ctx_t::out(const std::string &name, const duration_t &duration)
{
    total_durations[name] += duration;
    ++call_times[name];
    --depth;
}

void tracer_ctx_t::indent()
{
    for (int i = 0; i < depth; ++i) { printf("    "); }
}

tracer_ctx_t default_tracer_ctx("global");

tracer_t::tracer_t(const std::string &name, tracer_ctx_t &ctx)
    : name(name), t0(std::chrono::system_clock::now()), ctx(ctx)
{
    ctx.indent();
    WITH_XTERM(1, 35, printf("{ // [%s]", name.c_str()));
    putchar('\n');
    ctx.in();
}

tracer_t::~tracer_t()
{
    const auto now = std::chrono::system_clock::now();
    const std::chrono::duration<double> d = now - t0;
    ctx.out(name, d);
    char buffer[128];
    sprintf(buffer, "[%s] took %fs", name.c_str(), d.count());
    ctx.indent();
    WITH_XTERM(1, 32, printf("} // %s", buffer));
    putchar('\n');
}
