#pragma once
#include <chrono>
#include <cstdio>
#include <string>

#include <crystalnet/core/error.hpp>

struct tracer_ctx_t {
    int depth;

    void in() { ++depth; }
    void out() { --depth; }
    void indent();

    // template <typename... Args> void logf(const Args &... args)
    // {
    //     check(depth > 0);
    //     indent();
    //     printf("// ");
    //     printf(args...);
    //     putchar('\n');
    // }
};

struct tracer_t {
    const std::string name;
    const std::chrono::time_point<std::chrono::system_clock> t0;
    tracer_ctx_t &ctx;

    tracer_t(const std::string &, tracer_ctx_t &);
    ~tracer_t();
};

extern tracer_ctx_t default_tracer_ctx;

#define TRACE(name) tracer_t _((name), default_tracer_ctx)

#define _TRACE_WITH_NAMD(name, e)                                              \
    {                                                                          \
        tracer_t _(name, default_tracer_ctx);                                  \
        e;                                                                     \
    }

#define TRACE_IT(e) _TRACE_WITH_NAMD(#e, e);

#define TRACE_NAME(name, e) _TRACE_WITH_NAMD(std::string(#e "::") + name, e);
