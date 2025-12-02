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

template <typename TR, typename TA, typename TB>
struct CPUArithmeticSingularValues {
  // needed to work correctly
  static_assert(
      std::is_same_v<typename TR::FloatType, typename TA::FloatType> &&
      std::is_same_v<typename TR::FloatType, typename TB::FloatType>);

  using T = TR::FloatType;

  static constexpr u32 F32_EXP_SHIFT = 23;
  static constexpr u32 F32_SIGN_SHIFT = 31;
  static constexpr u32 F32_EXP_BIAS = 127;
  static constexpr u32 fracMask32 = ((u32)1 << F32_EXP_SHIFT) - 1;
  static constexpr u32 expShift32 = F32_EXP_SHIFT;
  static constexpr u32 expMask32 =
      (((u32)1 << (F32_SIGN_SHIFT - F32_EXP_SHIFT)) - 1) << F32_EXP_SHIFT;
  static constexpr u32 signShift32 = F32_SIGN_SHIFT;
  static constexpr u32 signMask32 = ((u32)1 << F32_SIGN_SHIFT);
  static constexpr u32 fracExpMask32 = fracMask32 ^ expMask32;

  static constexpr u32 fracMask = ((u32)1 << T::SignificandBits()) - 1;
  static constexpr u32 expShift = T::SignificandBits();
  static constexpr u32 expMask = (((u32)1 << T::ExponentBits()) - 1)
                                 << expShift;
  static constexpr u32 signShift = expShift + T::ExponentBits();
  static constexpr u32 signMask = ((u32)1 << signShift);
  static constexpr u32 fracExpMask = fracMask ^ expMask;

  static constexpr u32 expDiff = (F32_EXP_BIAS - T::BiasValue())
                                 << F32_EXP_SHIFT;

  static_assert(expShift <= F32_EXP_SHIFT);
  static_assert(signShift <= F32_SIGN_SHIFT);

  static inline void AddAt(TR &r, const u16 rIdx, const TA &a, const u16 aIdx,
                           const TB &b, const u16 bIdx) {
    AnyOpAt<AddOp>(r, rIdx, a, aIdx, b, bIdx);
  }

  static inline void SubAt(TR &r, const u16 rIdx, const TA &a, const u16 aIdx,
                           const TB &b, const u16 bIdx) {
    AnyOpAt<SubOp>(r, rIdx, a, aIdx, b, bIdx);
  }

  static inline void MulAt(TR &r, const u16 rIdx, const TA &a, const u16 aIdx,
                           const TB &b, const u16 bIdx) {
    AnyOpAt<MulOp>(r, rIdx, a, aIdx, b, bIdx);
  }

  static inline void DivAt(TR &r, const u16 rIdx, const TA &a, const u16 aIdx,
                           const TB &b, const u16 bIdx) {
    AnyOpAt<DivOp>(r, rIdx, a, aIdx, b, bIdx);
  }

  template <OperationType op>
  static inline f32 toMagicValue(u32 aBits, auto aBias, auto rBias) {
    u32 aFracExp = aBits & fracExpMask;
    u32 aSign = aBits & signMask;
    u32 aU32 = 0;
    aU32 ^= aSign << (F32_SIGN_SHIFT - signShift);
    aU32 ^= aFracExp << (F32_EXP_SHIFT - expShift);
    if constexpr (op == AddOp || op == SubOp)
      aU32 -= (rBias - aBias) << F32_EXP_SHIFT;
    else
      aU32 += expDiff;
    return std::bit_cast<f32>(aU32);
  }

  template <OperationType op>
  static inline u32 fromMagicValue(f32 rF32, auto rBias, auto aBias,
                                   auto bBias) {
    u32 rU32 = std::bit_cast<u32>(rF32);
    if constexpr (op == MulOp || op == DivOp) {
      u32 deltaBias = rBias - (op == MulOp ? aBias + bBias : aBias - bBias);
      u32 expDelta = expDiff + (deltaBias << F32_EXP_SHIFT);
      if ((rU32 & expMask32) >= expDelta)
        rU32 -= expDelta;
      else
        rU32 = 0;
    }
    u32 rFracExp = rU32 & fracExpMask32;
    u32 rSign = rU32 & signMask32;
    u32 rBits = 0;
    rBits ^= rSign >> (F32_SIGN_SHIFT - signShift);
    rBits ^= rFracExp >> (F32_EXP_SHIFT - expShift);
    assert((rFracExp >> (F32_EXP_SHIFT - expShift)) <
           signMask); // Exponent OF otherwise
    return rBits;
  }

  // NOTE: the bias is passed directly since it is expensive to calculate
  //       and remains always the same in consecutive calls!
  template <OperationType op>
  static inline void AnyOpAt(TR &r, const u16 rIdx, const TA &a, const u16 aIdx,
                             const TB &b, const u16 bIdx) {
    auto rBias = r.ScalarBits();
    auto aBias = a.ScalarBits();
    auto bBias = b.ScalarBits();
    f32 aF32 =
        toMagicValue<op>(a.template AtUnsafeBits<u32>(aIdx), aBias, rBias);
    f32 bF32 =
        toMagicValue<op>(b.template AtUnsafeBits<u32>(bIdx), aBias, rBias);
    f32 rF32;
    if constexpr (op == AddOp)
      rF32 = aF32 + bF32;
    else if constexpr (op == SubOp)
      rF32 = aF32 - bF32;
    else if constexpr (op == MulOp)
      rF32 = aF32 * bF32;
    else if constexpr (op == DivOp)
      rF32 = aF32 / bF32;
    else
      static_assert(false);
    u32 rBits = fromMagicValue<op>(rF32, rBias, aBias, bBias);
    r.SetBitsAtUnsafe(rIdx, rBits);
  }
};
#endif // BFPMX_CPU_ARITHMETIC_SINGULAR_VALUE_H
