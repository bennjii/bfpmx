//
// Created by Benjamin White on 24/10/2025.
//

#ifndef BFPMX_QUANTIZE_H
#define BFPMX_QUANTIZE_H

#include "arch/prelude.h"

template <template <std::size_t, BlockDimsType, IFloatRepr, template <typename> typename ArithmeticPolicy_> typename QuantizationPolicy,
          std::size_t ScalarBytes,
          typename BlockShape, // Concept can not be constrained
          typename Float, template <typename> typename ArithmeticPolicy>
concept IQuantize =
    IFloatRepr<Float> && BlockDimsType<BlockShape> &&
    requires(std::array<f64, BlockShape::TotalSize()> &v,
             Block<ScalarBytes, BlockShape, Float, ArithmeticPolicy, QuantizationPolicy> &b) {
      {
        QuantizationPolicy<ScalarBytes, BlockShape, Float, ArithmeticPolicy>::Quantize(v)
      } -> std::convertible_to<
          Block<ScalarBytes, BlockShape, Float, ArithmeticPolicy, QuantizationPolicy>>;

      {
        QuantizationPolicy<ScalarBytes, BlockShape, Float, ArithmeticPolicy>::Identity()
      } -> std::convertible_to<std::string>;
    };

#endif // BFPMX_QUANTIZE_H/