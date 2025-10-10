//
// Created by Benjamin White on 10/10/2025.
//

#include <array>
#include <cstddef>
#include "alias.cpp"
#include "block_float/repr/FloatRepr.h"

// template<typename T>
// concept IBlockFloat = requires(const T& s) {
//     { s.bits_of_significand() } -> std::convertible_to<i16>;
//     { s.bits_of_exponent() } -> std::convertible_to<i16>;
//     { s.bits_of_sign() } -> std::convertible_to<i16>;
//     { s.bits_of_scalar() } -> std::convertible_to<i16>;
// };
