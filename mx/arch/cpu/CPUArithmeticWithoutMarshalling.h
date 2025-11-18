//
// Created by Gabr1313 on 18/11/2025.
//

#ifndef BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H 
#define BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H

#include "definition/alias.h"
#include <cmath>
#include <iostream>
#include <type_traits>

template <typename T> struct CPUArithmeticWithoutMarshalling {
  // TODO: should this be an alternative in the policy? How to do so?
  static auto Add(const T &lhs, const T &rhs) -> T {
    // NOTE: should we lower those to u16 and i16? It depends on what kind of
    // values we pass...
    //       but this would be usefull in SIMD
    // TODO: what is max(expShift) given iT?
    using iT = i16;

    const auto aB = lhs.ScalarBits();
    const auto bB = rhs.ScalarBits();
    const auto rB = std::max(aB, bB) + 1;

    const iT fracMask = (1ull << T::FloatType::SignificandBits()) - 1;
    const iT expShift = T::FloatType::SignificandBits();
    const iT expMask = (1ull << T::FloatType::ExponentBits()) - 1;
    const iT signShift = expShift + T::FloatType::ExponentBits();
    const iT signMaskReal = (1ull << signShift);

    T result{T::Uninitialized};
    result.SetScalar(rB); // NOTE +1 so that I can put a -1 in the exponent of
                          // r[i] to avoid OF
    for (size_t i = 0; i < T::Length(); i++) {
      auto aPacked = lhs.AtUnsafe(i);
      auto bPacked = rhs.AtUnsafe(i);
      iT aBits = 0, bBits = 0;
      for (size_t j = 0; j < aPacked.size(); ++j) {
        aBits |= static_cast<iT>(aPacked[j]) << (8 * j);
        bBits |= static_cast<iT>(bPacked[j]) << (8 * j);
      }
      const iT aE = (aBits >> expShift) & expMask;
      const iT bE = (bBits >> expShift) & expMask;
      const iT deltaE = aE + aB - bE - bB;
      iT aM = 0, bM = 0;
      // TODO: Load subnormal values (not fully covered by the implementation
      //       of RealizeAtUnsafe(), Marshal/UnMarshal yet)
      // Here I shift everything by +expShift in order to not occur in precision
      // problems... if value is not a zero => construct it
      if (aBits & ~signMaskReal)
        aM = ((1ull << expShift) + (aBits & fracMask))
             << (expShift + (deltaE >= 0 ? +deltaE : 0));
      if (bBits & ~signMaskReal)
        bM = ((1ull << expShift) + (bBits & fracMask))
             << (expShift + (deltaE >= 0 ? 0 : -deltaE));
      if (aBits & signMaskReal)
        aM = -aM;
      if (bBits & signMaskReal)
        bM = -bM;
      iT rM = aM + bM;
      iT isNegative = rM < 0;
      if (isNegative)
        rM = -rM;
      const iT rDeltaE =
          sizeof(iT) * 8 - 1 - std::countl_zero((unsigned iT)rM) - expShift;
      if (rDeltaE >= 0)
        rM >>= rDeltaE;
      else
        rM <<= -rDeltaE;
      const iT exponent =
          (deltaE >= 0 ? bB + bE : aB + aE) - rB + rDeltaE - expShift;
      iT rBits = 0;
      if (exponent > 0) {
        rBits = (isNegative << signShift) | (exponent << expShift) |
                (rM & fracMask);
      } else { // subnormal values
        // TODO: Store subnormal values (not fully covered by the implementation
        //       of RealizeAtUnsafe(), Marshal/UnMarshal yet)
        // rBits = (isNegative << signShift)
        //       | ((rM >> -exponent) & fracMask);
      }
      result.SetBitsAt(i, rBits);
    }
    return result;
  }

  static auto Sub(const T &lhs, const T &rhs) -> T {
    // TODO
    using ElemType = f64;
    std::array<ElemType, T::dataCount()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::dataCount(); ++i)
      result[i] = l[i] - r[i];

    return T(result);
  }

  static auto Mul(const T &lhs, const T &rhs) -> T {
    // TODO
    using ElemType = f64;
    std::array<ElemType, T::dataCount()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::dataCount(); ++i)
      result[i] = l[i] * r[i];

    return T(result);
  }

  static auto Div(const T &lhs, const T &rhs) -> T {
    // TODO
    using ElemType = f64;
    std::array<ElemType, T::dataCount()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::dataCount(); ++i)
      result[i] = l[i] / r[i];

    return T(result);
  }
};

#endif // BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H
