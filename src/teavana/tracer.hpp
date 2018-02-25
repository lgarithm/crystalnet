#pragma once

#include <chrono>   // for duration, system_clock, time_point
#include <cstddef>  // for size_t
#include <cstdint>  // for uint32_t, uint64_t, uint8_t
#include <cstdio>   // for printf, putchar
#include <string>   // for string
#include <typeinfo> // for type_info

#include "teavana/range.hpp" // for range
#include "teavana/show.hpp"  // for size2str
#include "teavana/timer.hpp" // for timer
#include "teavana/core/shape.hpp"

namespace tea
{
struct tensor_tracer {
    static uint64_t total_usage;
    static uint32_t total_count;
    static bool log_on;

    static void off() { log_on = false; }
    static void toggole() { log_on ^= true; }

    template <typename R, uint8_t r> static void log_usage(const shape_t<r> &s)
    {
        size_t size = dim(s) * sizeof(R);
        if (log_on) {
            timer::logf("new tensor :: %s [[%lu]] (%s)", to_str(s).c_str(),
                        size, size2str(size).c_str());
        }
        total_usage += size;
        total_count += 1;
    }

    const ::std::string name;
    const uint64_t before_usage;
    const uint32_t before_count;
    tensor_tracer(const ::std::string &name = "")
        : name(name), before_usage(total_usage), before_count(total_count)
    {
    }

#ifdef ENABLE_TRACE_TENSOR_USAGE
    ~tensor_tracer()
    {
        uint64_t scope_usage = total_usage - before_usage;
        uint32_t scope_count = total_count - before_count;
        timer::logf("[%s] requested %ld tensors, total size [[%" PRId64
                    "]] (%s)",
                    name.c_str(), scope_count, scope_usage,
                    size2str(scope_usage).c_str());
    }
#endif
};

// uint64_t tensor_tracer::total_usage = 0;
// uint32_t tensor_tracer::total_count = 0;
// bool tensor_tracer::log_on = true;

#ifdef ENABLE_TRACE_TENSOR_USAGE
#define TRACE_TENSOR_USAGE(R, s) tensor_tracer::log_usage<R>(s);
#else
#define TRACE_TENSOR_USAGE(R, s)
#endif

inline void counter_report_header()
{
    printf("%-16s %-16s %-16s %s\n", "count", "time (s)", "mean (us)", "name");
}

struct emit_header_once {
    bool done = false;

    void operator()()
    {
        if (done) {
            return;
        }
        done = true;
        counter_report_header();
        for (auto _ : range(80)) {
            putchar('-');
        }
        putchar('\n');
    }
};

#ifdef ENABLE_PROFILE

emit_header_once emit_header;

#endif

template <typename T> struct count_tracer {
    static constexpr const char *name = T::name;

    static uint32_t count;
    static ::std::chrono::duration<double> time;

    static void reset()
    {
        count = 0;
        time = ::std::chrono::duration<double>(0);
    }

    static void report()
    {
        auto t = time.count();
        printf("%-16u %-16f %-16f %s\n", count, t, t * 1e6 / count, name);
    }

    const ::std::chrono::time_point<::std::chrono::system_clock> t0;

    count_tracer() : t0(::std::chrono::system_clock::now()) {}

    ~count_tracer()
    {
        auto now = ::std::chrono::system_clock::now();
        ::std::chrono::duration<double> d = now - t0;
        ++count;
        time += d;
    }
};

template <typename T> uint32_t count_tracer<T>::count = 0;
template <typename T>
::std::chrono::duration<double>
    count_tracer<T>::time = ::std::chrono::duration<double>(0);

template <typename Op, typename trait> struct op_count_tracer;

template <typename Op, typename trait> struct op_counter {
    static uint32_t count;
    static ::std::chrono::duration<double> time;

    ~op_counter()
    {
#ifdef ENABLE_PROFILE
        emit_header();
#endif
        // ::std::string name = Op::name;
        ::std::string name = typeid(Op).name();
        name += "::";
        name += trait::name;
        op_count_tracer<Op, trait>::report(name);
    }
};

template <typename Op, typename T> uint32_t op_counter<Op, T>::count = 0;
template <typename Op, typename T>
::std::chrono::duration<double>
    op_counter<Op, T>::time = ::std::chrono::duration<double>(0);

template <typename Op, typename trait> struct op_count_tracer {
    static op_counter<Op, trait> r;

    static void report(const ::std::string &name = "")
    {
        const auto t = r.time.count();
        printf("%-16u %-16f %-16f %s\n", r.count, t, t * 1e6 / r.count,
               name.c_str());
    }

    const ::std::chrono::time_point<::std::chrono::system_clock> t0;

    op_count_tracer() : t0(::std::chrono::system_clock::now()) {}

    ~op_count_tracer()
    {
        auto now = ::std::chrono::system_clock::now();
        ::std::chrono::duration<double> d = now - t0;
        ++r.count;
        r.time += d;
    }
};

template <typename Op, typename T> op_counter<Op, T> op_count_tracer<Op, T>::r;

#ifdef ENABLE_PROFILE

#define DEF_COUNTER(_name)                                                     \
    struct counter_type_##_name {                                              \
        static constexpr const char *name = #_name;                            \
    };                                                                         \
    struct counter_reporter_##_name {                                          \
        ~counter_reporter_##_name()                                            \
        {                                                                      \
            emit_header();                                                     \
            count_tracer<counter_type_##_name>::report();                      \
        }                                                                      \
    };                                                                         \
    counter_reporter_##_name run_on_exit_report_##_name;

#define SET_COUNTER(_name)                                                     \
    count_tracer<counter_type_##_name> _count_tracer_instance;

#else

#define DEF_COUNTER(_name)
#define SET_COUNTER(_name)

#endif
}
