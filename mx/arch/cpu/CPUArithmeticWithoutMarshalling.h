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
  static auto Add(const T &lhs, const T &rhs) -> T {
    // NOTE: should we lower those to i8 or i16? It depends on what kind of
    //       values we pass... (see NOTE.2)
    //       but this would be usefull in SIMD
    using iT = i64;

    const auto aBias = lhs.ScalarBits();
    const auto bBias = rhs.ScalarBits();
    const auto rBias =
        std::max(aBias, bBias) + 1; // +1 so that I can avoid OF
                                    // n_bits + n_bits = (n+1)_bits

    const iT fracMask = (1ull << T::FloatType::SignificandBits()) - 1;
    const iT expShift = T::FloatType::SignificandBits();
    const iT expMask = (1ull << T::FloatType::ExponentBits()) - 1;
    const iT signShift = expShift + T::FloatType::ExponentBits();
    const iT signMaskReal = (1ull << signShift);

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
      const iT aExp = (aBits >> expShift) & expMask;
      const iT bExp = (bBits >> expShift) & expMask;
      const iT deltaExp = aExp + aBias - bExp - bBias;
      iT aSignif = 0, bSignif = 0;
      // TODO: Load subnormal values

      // NOTE.2: alternative: it uses less bits and it is what is implemented in
      //       HW, but it looses a bit precision
      //       !! remember to also change the line with
      //       `NOTE.2: alternative: see above` ~30 lines down
      // if (aBits & ~signMaskReal)
      //   aSignif = ((1ull << expShift) + (aBits & fracMask)) >>
      //        (deltaExp >= 0 ? 0 : -deltaExp);
      // if (bBits & ~signMaskReal)
      //   bSignif = ((1ull << expShift) + (bBits & fracMask)) >>
      //        (deltaExp >= 0 ? +deltaExp : 0);

      if (aBits & ~signMaskReal)
        aSignif = ((1ull << expShift) + (aBits & fracMask))
                  << (deltaExp >= 0 ? +deltaExp : 0);
      if (bBits & ~signMaskReal)
        bSignif = ((1ull << expShift) + (bBits & fracMask))
                  << (deltaExp >= 0 ? 0 : -deltaExp);
      if (aBits & signMaskReal)
        aSignif = -aSignif;
      if (bBits & signMaskReal)
        bSignif = -bSignif;
      iT rSigif = aSignif + bSignif;
      iT isNegative = rSigif < 0;
      if (isNegative)
        rSigif = -rSigif;
      const iT rDeltaE =
          sizeof(rSigif) * 8 - 1 - expShift -
          std::countl_zero(static_cast<std::make_unsigned_t<iT>>(rSigif));
      if (rDeltaE >= 0)
        rSigif >>= rDeltaE;
      else
        rSigif <<= -rDeltaE;
      const iT exponent =
          (deltaExp >= 0 ? bBias + bExp : aBias + aExp) - rBias + rDeltaE;
      // NOTE.2: alternative: see above
      // const iT exponent =
      //   (deltaExp >= 0 ? aBias + aExp : bBias + bExp) - rBias + rDeltaE;
      iT rBits = 0;
      if (exponent > 0 && rSigif) {
        rBits = (isNegative << signShift) | (exponent << expShift) |
                (rSigif & fracMask);
      } else { // TODO: store subnormal values
        // rBits = (isNegative << signShift)
        //       | ((rSigif >> -exponent) & fracMask);
      }
      result.SetBitsAtUnsafe(i, rBits);
    }
    return result;
  }

  static auto Sub(const T &lhs, const T &rhs) -> T {
    // TODO
    return lhs;
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
