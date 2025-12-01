//
// Created by Gabr1313 on 18/11/2025.
//
#ifndef BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H
#define BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H
#include "definition/alias.h"
#include <cassert>

#include "CPUArithmeticSingularValues.h"

template <typename T, template <typename, typename, typename> class Arithmetic =
                          CPUArithmeticSingularValues>
struct CPUArithmeticWithoutMarshalling {

  using iT = i64;
  static auto Add(const T &a, const T &b) -> T { return _AnyOp<AddOp>(a, b); }

  static auto Sub(const T &a, const T &b) -> T { return _AnyOp<SubOp>(a, b); }

  static auto Mul(const T &a, const T &b) -> T { return _AnyOp<MulOp>(a, b); }

  static auto Div(const T &a, const T &b) -> T { return _AnyOp<DivOp>(a, b); }

  static constexpr iT fracMask = (1ull << T::FloatType::SignificandBits()) - 1;
  static constexpr iT expShift = T::FloatType::SignificandBits();
  static constexpr iT expMask = (1ull << T::FloatType::ExponentBits()) - 1;
  static constexpr iT signShift = expShift + T::FloatType::ExponentBits();
  static constexpr iT signMask = (1ull << signShift);

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

    T result{typename T::Uninitialized{}};
    result.SetScalar(rBias);
    for (std::size_t i = 0; i < T::Length(); i++) {
      Arithmetic<T, T, T>::template AnyOpAt<op>(result, i, rBias, a, i, aBias,
                                                b, i, bBias);
    }
    return result;
  }
};
#endif // BFPMX_CPU_ARITHMETIC_WITHOUT_MARSHALLING_H
