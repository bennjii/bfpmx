#ifndef BFPMX_UTIL_C
#define BFPMX_UTIL_C

#include "prelude.h"

constexpr f64 LEEWAY = 2.0;

template <IFloatRepr Float>
bool FuzzyEqual(const f64 a, const f64 b, const f64 leeway = LEEWAY) {
  return std::abs(a - b) <=
         leeway * Float::Epsilon() *
             std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

#endif // BFPMX_UTIL_C
