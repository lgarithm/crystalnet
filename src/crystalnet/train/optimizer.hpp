#pragma once
#include <crystalnet.h>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/model/model.hpp>

struct optimizer_ctx_t {
    virtual void operator()() = 0;
};

struct optimizer_t {
    virtual optimizer_ctx_t *optimize(model_t *) = 0;
};
