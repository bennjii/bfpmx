//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_CPU_ARITHMETIC_H
#define BFPMX_CPU_ARITHMETIC_H

#include "definition/alias.h"

#include <type_traits>

template <typename T> struct CPUArithmetic {
  static auto Add(const T &lhs, const T &rhs) -> T {
    using ElemType = f64;
    std::array<ElemType, T::Length()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::Length(); ++i)
      result[i] = l[i] + r[i];

    return T(result);
  }

  static auto Sub(const T &lhs, const T &rhs) -> T {
    using ElemType = f64;
    std::array<ElemType, T::dataCount()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::dataCount(); ++i)
      result[i] = l[i] - r[i];

    return T(result);
  }

  static auto Mul(const T &lhs, const T &rhs) -> T {
    using ElemType = f64;
    std::array<ElemType, T::dataCount()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::dataCount(); ++i)
      result[i] = l[i] * r[i];

    return T(result);
  }

  static auto Div(const T &lhs, const T &rhs) -> T {
    using ElemType = f64;
    std::array<ElemType, T::dataCount()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::dataCount(); ++i)
      result[i] = l[i] / r[i];

    return T(result);
  }
};

#endif // BFPMX_CPU_ARITHMETIC_H