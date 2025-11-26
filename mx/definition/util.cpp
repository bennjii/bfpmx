#ifndef BFPMX_UTIL_C
#define BFPMX_UTIL_C

#include "prelude.h"

constexpr f64 LEEWAY = 2.0;

bool FuzzyEqual(const f64 a, const f64 b, const f64 maxDiff) {
    return std::abs(a - b) <= maxDiff;
}

template <IFloatRepr Float>
bool FuzzyEqual(const f64 a, const f64 b, const f64 leeway = LEEWAY) {
    return FuzzyEqual(a, b, Float::Epsilon() * leeway);
}

#endif // BFPMX_UTIL_C
