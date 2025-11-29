//
// Created by Gabr1313 on 18/11/2025.
//
#ifndef BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H
#define BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H
#include "definition/alias.h"
#include <cassert>

#include "CPUArithmeticSingularValues.h"

template <typename T, typename iT = i64>
struct CPUArithmeticWithoutMarshalling {

  static auto Add(const T &a, const T &b) -> T { return _AnyOp<AddOp>(a, b); }

  static auto Sub(const T &a, const T &b) -> T { return _AnyOp<SubOp>(a, b); }

  static auto Mul(const T &a, const T &b) -> T { return _AnyOp<MulOp>(a, b); }

  static auto Div(const T &a, const T &b) -> T { return _AnyOp<DivOp>(a, b); }

  static const iT fracMask = (1ull << T::FloatType::SignificandBits()) - 1;
  static const iT expShift = T::FloatType::SignificandBits();
  static const iT expMask = (1ull << T::FloatType::ExponentBits()) - 1;
  static const iT signShift = expShift + T::FloatType::ExponentBits();
  static const iT signMask = (1ull << signShift);

  template <OperationType op> static auto _AnyOp(const T &a, const T &b) -> T {
    const auto aBias = a.ScalarBits();
    const auto bBias = b.ScalarBits();
    auto rBias = 0;
    if constexpr (op == AddOp || op == SubOp)
      rBias = std::max(aBias, bBias);
    else if constexpr (op == MulOp)
      rBias = aBias + bBias;
    else if constexpr (op == DivOp)
      rBias = aBias - bBias;
    else
      static_assert(false);

    using UninitTag = typename T::Uninitialized;
    T result{UninitTag{}};
    result.SetScalar(rBias);
    for (std::size_t i = 0; i < T::Length(); i++) {
      if constexpr (op == AddOp || op == SubOp)
        CPUArithmeticSingularValues<T, T, T, iT>::template _AddOrSubAt<op>(
            result, i, rBias, a, i, aBias, b, i, bBias);
      else
        CPUArithmeticSingularValues<T, T, T, iT>::template _MulOrDivAt<op>(
            result, i, rBias, a, i, aBias, b, i, bBias);
    }
    return result;
  }
};
#endif // BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H
