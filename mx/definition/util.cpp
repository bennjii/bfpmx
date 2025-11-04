#ifndef BFPMX_UTIL_C
#define BFPMX_UTIL_C

#include "prelude.h"

constexpr f64 LEEWAY = 2.0f;

template <IFloatRepr Float> bool FuzzyEqual(const f64 a, const f64 b) {
  return std::abs(a - b) <= LEEWAY * Float::Epsilon();
}

#endif // BFPMX_UTIL_C