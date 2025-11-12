//
// Created by Benjamin White on 24/10/2025.
//

#ifndef BFPMX_QUANTIZE_H
#define BFPMX_QUANTIZE_H

#include "arch/prelude.h"

template <template <std::size_t, BlockDimsType, IFloatRepr> typename T,
          std::size_t ScalarBytes,
          typename BlockShape, // Concept can not be constrained
          typename Float>
concept IQuantize =
    IFloatRepr<Float> && BlockDimsType<BlockShape> &&
    requires(std::array<f64, BlockShape::TotalSize()> &v,
             Block<ScalarBytes, BlockShape, Float, CPUArithmetic, T> &b) {
      {
        T<ScalarBytes, BlockShape, Float>::Quantize(v)
      } -> std::convertible_to<
          Block<ScalarBytes, BlockShape, Float, CPUArithmetic, T>>;
    };

#endif // BFPMX_QUANTIZE_H/