//
// Created by Gabr1313 on 18/11/2025.
//

#ifndef BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H
#define BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H

#include "definition/alias.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>

template <typename T> struct CPUArithmeticWithoutMarshalling {
  // NOTE: can easily become branchless
  // NOTE: should we lower those to i8 or i16?
  //       It depends on what kind of values we pass...
  //       We need at least expShift*2+1 bits
  using iT = i64;
  static const iT fracMask = (1ull << T::FloatType::SignificandBits()) - 1;
  static const iT expShift = T::FloatType::SignificandBits();
  static const iT expMask = (1ull << T::FloatType::ExponentBits()) - 1;
  static const iT signShift = expShift + T::FloatType::ExponentBits();
  static const iT signMaskReal = (1ull << signShift);

  template <typename Op>
  static auto _AddOrSub(const T &lhs, const T &rhs, Op op) -> T {
    const auto aBias = lhs.ScalarBits();
    const auto bBias = rhs.ScalarBits();
    const auto rBias = std::max(aBias, bBias);

    T result{T::Uninitialized};
    result.SetScalar(rBias);
    for (size_t i = 0; i < T::Length(); i++) {
      auto aPacked = lhs.AtUnsafe(i);
      auto bPacked = rhs.AtUnsafe(i);
      iT aBits = 0, bBits = 0;
      for (size_t j = 0; j < aPacked.size(); ++j) {
        aBits |= static_cast<iT>(aPacked[j]) << (8 * j);
        bBits |= static_cast<iT>(bPacked[j]) << (8 * j);
      }

      iT aExp = (aBits >> expShift) & expMask;
      iT bExp = (bBits >> expShift) & expMask;
      const iT deltaExpAB = aExp + aBias - bExp - bBias;
      iT deltaExpShift;

      iT aSignif = 0, bSignif = 0;
      if (aBits & ~signMaskReal)
        aSignif = ((1ull << expShift) + (aBits & fracMask));
      if (bBits & ~signMaskReal)
        bSignif = ((1ull << expShift) + (bBits & fracMask));
      iT shift = 0;
      if (deltaExpAB >= 0) {
        shift = std::min(deltaExpAB, expShift + 0);
        aSignif <<= shift;
        bSignif >>= deltaExpAB - shift;
      } else {
        shift = std::min(-deltaExpAB, expShift + 0);
        aSignif >>= -deltaExpAB - shift;
        bSignif <<= shift;
      }
      deltaExpShift = shift + rBias;
      if (aBits & signMaskReal)
        aSignif = -aSignif;
      if (bBits & signMaskReal)
        bSignif = -bSignif;
      iT rSignif = op(aSignif, bSignif);
      iT isNegative = rSignif < 0;
      if (isNegative)
        rSignif = -rSignif;
      const iT rDeltaExp =
          sizeof(rSignif) * 8 - 1 - expShift -
          std::countl_zero(static_cast<std::make_unsigned_t<iT>>(rSignif));
      if (rDeltaExp >= 0)
        rSignif >>= rDeltaExp;
      else
        rSignif <<= -rDeltaExp;
      const iT exponent = (deltaExpAB >= 0 ? aBias + aExp : bBias + bExp) -
                          deltaExpShift + rDeltaExp;
      iT rBits = 0;
      if (exponent > 0 && rSignif) {
        rBits = (isNegative << signShift) | (exponent << expShift) |
                (rSignif & fracMask);
      }
      result.SetBitsAtUnsafe(i, rBits);
    }
    return result;
  }

  static auto Add(const T &lhs, const T &rhs) -> T {
    return _AddOrSub(lhs, rhs, [](iT a, iT b) { return a + b; });
  }

  static auto Sub(const T &lhs, const T &rhs) -> T {
    return _AddOrSub(lhs, rhs, [](iT a, iT b) { return a - b; });
  }

  static auto Mul(const T &lhs, const T &rhs) -> T {
    // TODO
    return lhs;
  }

  static auto Div(const T &lhs, const T &rhs) -> T {
    // TODO
    return lhs;
  }
};

#endif // BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H
