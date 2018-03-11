#pragma once
#include <cstdio>
#include <cstdlib>

#define EXIT(err) exit_err(err)

inline void exit_err(const char *err)
{
    perror(err);
    exit(1);
}

void runtime_check(bool, const char *const, const char *const, int);

#define check(e) runtime_check((e), #e, __FILE__, __LINE__);
