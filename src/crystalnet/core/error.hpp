#pragma once
#include <cstdio>
#include <cstdlib>

#define EXIT(err) exit_err(err)

inline void exit_err(const char *err)
{
    perror(err);
    exit(1);
}
