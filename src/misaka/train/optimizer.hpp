#pragma once
#include <misaka.h>
#include <misaka/core/tensor.hpp>
#include <misaka/model/model.hpp>

struct optimizer_ctx_t {
    virtual void operator()() = 0;
};

struct optimizer_t {
    virtual optimizer_ctx_t *optimize(model_t *) = 0;
};
