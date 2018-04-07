#include <crystalnet.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/train/opt_adam.hpp>
#include <crystalnet/train/opt_sgd.hpp>
#include <crystalnet/train/optimizer.hpp>

static auto gc = GC<optimizer_t>();

const optimizer_t *opt_sgd = gc(new sgd_optimizer_t);
const optimizer_t *opt_adam = gc(new adam_optimizer_t);
