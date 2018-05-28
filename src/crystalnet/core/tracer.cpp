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
    FILE *fp = fopen("trace.log", "w");
    report(fp);
    fclose(fp);
}

void tracer_ctx_t::report(FILE *fp) const
{
    using item_t = std::tuple<duration_t, uint32_t, std::string>;
    std::vector<item_t> list;
    for (const auto [name, duration] : total_durations) {
        list.push_back(item_t(duration, call_times.at(name), name));
    }
    std::sort(list.rbegin(), list.rend());

    const std::string hr(80, '-');
    fprintf(fp, "\tsummary of %s::%s\n", "tracer_ctx_t", name.c_str());
    fprintf(fp, "%s\n", hr.c_str());
    fprintf(fp, "%8s    %16s    %s\n", "count", "total duration", "name");
    fprintf(fp, "%s\n", hr.c_str());
    for (const auto &[duration, count, name] : list) {
        fprintf(fp, "%8d    %16fs    %s\n",  //
                count, duration.count(), name.c_str());
    }
}

void tracer_ctx_t::out(const std::string &name, const duration_t &duration)
{
    total_durations[name] += duration;
    ++call_times[name];
    --depth;
}

void tracer_ctx_t::indent(FILE *fp)
{
    for (int i = 0; i < depth; ++i) { fprintf(fp, "    "); }
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

set_trace_log_t::set_trace_log_t(const std::string &name, bool reuse,
                                 tracer_ctx_t &ctx)
    : ctx(ctx), name(name)
{
    FILE *fp = reuse  //
                   ? std::fopen(name.c_str(), "a")
                   : std::fopen(name.c_str(), "w");
    ctx.log_files.push_front(fp);
    ctx.indent();
    ctx.logf1(stdout, "start logging to %s", name.c_str());
}

set_trace_log_t::~set_trace_log_t()
{
    ctx.indent();
    ctx.logf1(stdout, "stop logging to file://%s", name.c_str());
    FILE *fp = ctx.log_files.front();
    ctx.log_files.pop_front();
    std::fclose(fp);
}
