#pragma once

#ifdef MISAKA_USE_CBLAS

#include <misaka/linag/cblas_impl.hpp>
template <typename T> using linag = cblas_impl<T>;

#else

#include <misaka/linag/plain_impl.hpp>
template <typename T> using linag = plain_impl<T>;

#endif
