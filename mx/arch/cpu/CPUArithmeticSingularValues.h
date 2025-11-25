//
// Created by Gabr1313 on 25/11/2025.
//
#ifndef BFPMX_CPU_ARITHMETIC_SINGULAR_VALUE_H
#define BFPMX_CPU_ARITHMETIC_SINGULAR_VALUE_H
#include "definition/alias.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <type_traits>

enum OperationType { AddOp, SubOp, MulOp, DivOp };

template <typename TR, typename TA, typename TB, typename iT = i64>
struct CPUArithmeticSingularValues {
  // NOTE: can easily become branchless
  // NOTE: should we lower those to i8 or i16?
  //       It depends on what kind of values we pass...
  //       We need at least expShift*2+2 bits

  // needed to work correctly
  static_assert(
      std::is_same_v<typename TR::FloatType, typename TA::FloatType> &&
      std::is_same_v<typename TR::FloatType, typename TB::FloatType>);

  using T = TR::FloatType;

  static const iT fracMask = ((iT)1 << T::SignificandBits()) - 1;
  static const iT expShift = T::SignificandBits();
  static const iT expMask = ((iT)1 << T::ExponentBits()) - 1;
  static const iT signShift = expShift + T::ExponentBits();
  static const iT signMask = ((iT)1 << signShift);

  static inline void AddAt(TR &r, const u16 rIdx, auto rBias, const TA &a,
                           const u16 aIdx, auto aBias, const TB &b,
                           const u16 bIdx, auto bBias) {
    _AddOrSubAt<AddOp>(r, rIdx, rBias, a, aIdx, aBias, b, bIdx, bBias);
  }

  static inline void SubAt(TR &r, const u16 rIdx, auto rBias, const TA &a,
                           const u16 aIdx, auto aBias, const TB &b,
                           const u16 bIdx, auto bBias) {
    _AddOrSubAt<SubOp>(r, rIdx, rBias, a, aIdx, aBias, b, bIdx, bBias);
  }

  static inline void MulAt(TR &r, const u16 rIdx, auto rBias, const TA &a,
                           const u16 aIdx, auto aBias, const TB &b,
                           const u16 bIdx, auto bBias) {
    _MulOrDivAt<MulOp>(r, rIdx, rBias, a, aIdx, aBias, b, bIdx, bBias);
  }

  static inline void DivAt(TR &r, const u16 rIdx, auto rBias, const TA &a,
                           const u16 aIdx, auto aBias, const TB &b,
                           const u16 bIdx, auto bBias) {
    _MulOrDivAt<DivOp>(r, rIdx, rBias, a, aIdx, aBias, b, bIdx, bBias);
  }

  // NOTE: the bias is passed directly since it is expensive to calculate
  //       and remains always the same in consecutive calls!
  template <OperationType op>
  static inline void _AddOrSubAt(TR &r, const u16 rIdx, auto rBias, const TA &a,
                                 const u16 aIdx, auto aBias, const TB &b,
                                 const u16 bIdx, auto bBias) {
    static_assert(op == AddOp || op == SubOp);
    auto aPacked = a.AtUnsafe(aIdx);
    auto bPacked = b.AtUnsafe(bIdx);
    iT aBits = 0, bBits = 0;
    for (std::size_t j = 0; j < aPacked.size(); ++j) {
      aBits |= static_cast<iT>(aPacked[j]) << (8 * j);
      bBits |= static_cast<iT>(bPacked[j]) << (8 * j);
    }
    iT aExp = (aBits >> expShift) & expMask;
    iT bExp = (bBits >> expShift) & expMask;
    const iT deltaExpAB = aExp + aBias - bExp - bBias;
    iT deltaExpShift;
    iT aSignif = 0, bSignif = 0;
    if (aBits & ~signMask) {
      aSignif = ((1ull << expShift) + (aBits & fracMask));
      if (aBits & signMask)
        aSignif = -aSignif;
    }
    if (bBits & ~signMask) {
      bSignif = ((1ull << expShift) + (bBits & fracMask));
      if (bBits & signMask)
        bSignif = -bSignif;
    }
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
    iT rSignif = (op == AddOp ? aSignif + bSignif : aSignif - bSignif);
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
    r.SetBitsAtUnsafe(rIdx, rBits);
  }

  // NOTE: the bias is passed directly since it is expensive to calculate
  //       and remains always the same in consecutive calls!
  template <OperationType op>
  static inline void _MulOrDivAt(TR &r, const u16 rIdx, auto rBias, const TA &a,
                                 const u16 aIdx, auto aBias, const TB &b,
                                 const u16 bIdx, auto bBias) {
    static_assert(op == MulOp || op == DivOp);
    auto aPacked = a.AtUnsafe(aIdx);
    auto bPacked = b.AtUnsafe(bIdx);
    iT aBits = 0, bBits = 0;
    for (std::size_t j = 0; j < aPacked.size(); ++j) {
      aBits |= static_cast<iT>(aPacked[j]) << (8 * j);
      bBits |= static_cast<iT>(bPacked[j]) << (8 * j);
    }
    iT aExp = (aBits >> expShift) & expMask;
    iT bExp = (bBits >> expShift) & expMask;
    iT aSignif = 0, bSignif = 0;
    if (aBits & ~signMask) {
      aSignif = ((1ull << expShift) + (aBits & fracMask));
      aSignif <<= op == MulOp ? 0 : expShift + 1;
    }
    if (bBits & ~signMask) {
      bSignif = ((1ull << expShift) + (bBits & fracMask));
    }
    iT rSignif = op == MulOp ? aSignif * bSignif : aSignif / bSignif;
    const iT rDeltaExp =
        sizeof(rSignif) * 8 - 1 - expShift -
        std::countl_zero(static_cast<std::make_unsigned_t<iT>>(rSignif));
    assert(!rSignif || rDeltaExp >= 0); // something went wrong
    rSignif >>= rDeltaExp;
    iT exponent =
        op == MulOp
            ? aExp + bExp - T::BiasValue() + (rDeltaExp != expShift)
            : aExp - bExp + T::BiasValue() - (rDeltaExp + 2 != expShift);
    iT rBits = 0;
    if (exponent > 0 && rSignif) {
      assert((exponent & expMask) == exponent); // OF otherwise
      iT rSign = (aBits & signMask) ^ (bBits & signMask);
      rBits = rSign | (exponent << expShift) | (rSignif & fracMask);
    }
    r.SetBitsAtUnsafe(rIdx, rBits);
  }
};
#endif // BFPMX_CPU_ARITHMETIC_SINGULAR_VALUE_H
