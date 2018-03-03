#pragma once

#ifdef CRYSTALNET_USE_CBLAS

#include <crystalnet/linag/cblas_impl.hpp>
template <typename T> using linag = cblas_impl<T>;

#else

#include <crystalnet/linag/plain_impl.hpp>
template <typename T> using linag = plain_impl<T>;

#endif
