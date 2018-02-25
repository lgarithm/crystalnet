#include <misaka.h>
#include <misaka/core/gc.hpp>
#include <misaka/train/optimizer.hpp>

static auto gc = GC<optimizer_t>();

optimizer_t *opt_sgd = gc(new optimizer_t);
