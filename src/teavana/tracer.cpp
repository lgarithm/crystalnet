#include <chrono>   // for duration, system_clock, time_point
#include <cstddef>  // for size_t
#include <cstdint>  // for uint32_t, uint64_t, uint8_t
#include <cstdio>   // for printf, putchar
#include <string>   // for string
#include <typeinfo> // for type_info

#include "teavana/show.hpp"  // for size2str
#include "teavana/timer.hpp" // for timer
#include "teavana/tracer.hpp"
#include <crystalnet/utility/range.hpp>

namespace tea
{
uint64_t tensor_tracer::total_usage = 0;
uint32_t tensor_tracer::total_count = 0;
bool tensor_tracer::log_on = true;

#ifdef ENABLE_TRACE_TENSOR_USAGE
#define TRACE_TENSOR_USAGE(R, s) tensor_tracer::log_usage<R>(s);
#else
#define TRACE_TENSOR_USAGE(R, s)
#endif

#ifdef ENABLE_PROFILE

emit_header_once emit_header;

#endif

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
