//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_CPU_ARITHMETIC_H
#define BFPMX_CPU_ARITHMETIC_H

#include "definition/alias.h"
#include <iostream>
#include <type_traits>

template <typename T> struct CPUArithmetic {
  using ElemType = f64;
  
  static auto Add(const T &lhs, const T &rhs) -> T {
    
    std::array<ElemType, T::Length()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::Length(); ++i)
      result[i] = l[i] + r[i];

    return T(result);
  }

  static auto Sub(const T &lhs, const T &rhs) -> T {
    std::array<ElemType, T::Length()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::Length(); ++i)
      result[i] = l[i] - r[i];

    return T(result);
  }

  static auto Mul(const T &lhs, const T &rhs) -> T {
    std::array<ElemType, T::Length()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::Length(); ++i)
      result[i] = l[i] * r[i];

    return T(result);
  }

  static auto Dot(const T &lhs, const T &rhs) -> f64 {
    ElemType result = lhs.Scalar() * rhs.Scalar(), elementSum = 0.;

    for (std::size_t i = 0; i < T::Length(); ++i)
      elementSum += lhs.ElementAt(i) * rhs.ElementAt(i);

    result *= elementSum;
    return result;
  }

  static auto Div(const T &lhs, const T &rhs) -> T {
    std::array<ElemType, T::Length()> result;

    auto l = lhs.Spread();
    auto r = rhs.Spread();

    for (std::size_t i = 0; i < T::Length(); ++i)
      result[i] = l[i] / r[i];

    return T(result);
  }

  template <typename MatrixBlockType, typename InT, typename OutT>
  static auto Gemv(const MatrixBlockType &matrix, const InT &vector) -> OutT {

    // safety checks
    static_assert(MatrixBlockType::NumDimensions == 2, "Matrix must be 2D");
    static_assert(InT::NumDimensions == 1, "Input vector must be 1D");
    static_assert(OutT::NumDimensions == 1, "Output vector must be 1D");

    // get dims
    const auto rows = MatrixBlockType::Dims[0];
    const auto cols = MatrixBlockType::Dims[1];
    const auto input_size = InT::Dims[0];
    const auto output_size = OutT::Dims[0];

    std::array<ElemType, rows> result;

    auto matrix_data = matrix.Spread();
    auto vec_data = vector.Spread();

    for (std::size_t i = 0; i < rows; i++) {
      ElemType sum = 0.0;
      for (std::size_t j = 0; j < cols; j++) {
        ElemType matrix_val = matrix_data[i * cols + j];
        ElemType vector_val = vec_data[j];
        ElemType product = matrix_val * vector_val;
        sum += product;
      }
      result[i] = sum;
    }
    return OutT(result);
  }

  template <typename MatAType, typename MatBType, typename OutT>
  static auto Gemm(const MatAType &matA, const MatBType &matB) -> OutT {
    // safety checks
    static_assert(MatAType::NumDimensions == 2, "Matrix A must be 2D");
    static_assert(MatBType::NumDimensions == 2, "matrix B must be 2D");
    static_assert(OutT::NumDimensions == 2, "Output must be 2D");

    // get dims
    const auto a_rows = MatAType::Dims[0];
    const auto a_cols = MatAType::Dims[1];
    const auto b_rows = MatBType::Dims[0];
    const auto b_cols = MatBType::Dims[1];
    const auto o_rows = OutT::Dims[0];
    const auto o_cols = OutT::Dims[1];

    std::array<ElemType, o_rows * o_cols> result;

    auto a_data = matA.Spread();
    auto b_data = matB.Spread();

    for (std::size_t i = 0; i < a_rows; i++) {
      for (std::size_t j = 0; j < b_cols; j++) {
        ElemType sum = 0.0;
        for (std::size_t k = 0; k < a_cols; k++) {
          ElemType a_val = a_data[i * a_cols + k]; // A[i][k]
          ElemType b_val = b_data[k * b_cols + j]; // B[k][j]
          sum += a_val * b_val;
        }
        result[i * b_cols + j] = sum; // C[i][j]
      }
    }
    return OutT(result);
  }
};

#endif // BFPMX_CPU_ARITHMETIC_H