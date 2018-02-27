#include <misaka.h>
#include <misaka/core/gc.hpp>
#include <misaka/train/opt_adam.hpp>
#include <misaka/train/opt_sgd.hpp>
#include <misaka/train/optimizer.hpp>

static auto gc = GC<optimizer_t>();

optimizer_t *opt_sgd = gc(new sgd_optimizer_t);
optimizer_t *opt_adam = gc(new adam_optimizer_t);
