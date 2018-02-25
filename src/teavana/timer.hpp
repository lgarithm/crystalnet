#pragma once

#include <chrono>  // for duration, system_clock, operator-, time_point
#include <cstdint> // for uint8_t
#include <cstdio>  // for putchar, printf, fclose, fopen, FILE, fprintf, etc
#include <ratio>   // for ratio
#include <string>  // for string

namespace tea
{
struct with_color {
    with_color(uint8_t b, uint8_t f) { printf("\e[%u;%um", b, f); }
    ~with_color() { printf("\e[m"); }
};

struct timer {
    static constexpr const char *log_file = "time.log";
    static bool on;
    static int dep;

    static void clear_log()
    {
        FILE *fp = fopen(log_file, "w");
        fclose(fp);
    }

    static void off() { on = false; }

    static void indent()
    {
        for (int i = 0; i < dep; ++i) {
            putchar(' ');
            putchar(' ');
            putchar(' ');
            putchar(' ');
        }
    }

    template <typename... Args> static void logf(const Args &... args)
    {
        if (on) {
            indent();
            printf("// ");
            printf(args...);
            putchar('\n');
        }
    }

    const ::std::chrono::time_point<::std::chrono::system_clock> t0;
    const ::std::string name;
    const bool write_file;

    timer(const ::std::string &name = "", bool write_file = false)
        : t0(::std::chrono::system_clock::now()), name(name),
          write_file(write_file)
    {
        if (on) {
            indent();
            {
                with_color _(1, 35);
                printf("{ // [%s]", name.c_str());
            }
            putchar('\n');
        }
        ++dep;
    }

    ~timer()
    {
        --dep;
        char buffer[128];
        {
            auto now = ::std::chrono::system_clock::now();
            ::std::chrono::duration<double> d = now - t0;
            sprintf(buffer, "[%s] took %fs", name.c_str(), d.count());
        }
        if (on) {
            indent();
            {
                with_color _(1, 32);
                printf("} // %s", buffer);
            }
            putchar('\n');
        }
        if (write_file) {
            FILE *fp = fopen(log_file, "a");
            fprintf(fp, "%s\n", buffer);
            fclose(fp);
        }
    }
};

#ifdef ENABLE_DEBUG_TIMER
#define DEBUG_TIMER(var, name) tea::timer var(name)
#else
#define DEBUG_TIMER(var, name)
#endif
}
