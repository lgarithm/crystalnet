#pragma once
#include <cstdint>
#include <crystalnet.h>
#include <string>

#include <teavana/timer.hpp>

#ifdef CRYSTALNET_DEBUG
#define DEBUG(func)                                                            \
    log_func_call(func);                                                       \
    DEBUG_TIMER(_, func)
#define LOG_TENSOR_USAGE(shape, size) log_tensor_usage(&shape, size)
#define LOG_NODE_USAGE(shape, name) log_node_usage(&shape, name)
#else
#define DEBUG(func)
#define LOG_TENSOR_USAGE(shape, size)
#define LOG_NODE_USAGE(shape, name)
#endif

void log_func_call(const char *);
void log_tensor_usage(const shape_t *, uint8_t);
void log_node_usage(const shape_t *, const std::string &);
