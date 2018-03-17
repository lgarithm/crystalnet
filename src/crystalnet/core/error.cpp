#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <crystalnet/core/error.hpp>

const char *get_src_name(const char *const file)
{
    int off = strlen(__FILE__) - strlen("crystalnet/model/node.hpp");
    return file + off;
}

void runtime_check(bool cond, const char *const e, const char *const file,
                   int line)
{
    if (!cond) {
        printf("contract not meet: %s:%d %s\n", get_src_name(file), line, e);
        exit(1);
    }
}
