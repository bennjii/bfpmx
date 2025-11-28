//
// Created by Benjamin White on 24/10/2025.
//

#ifndef BFPMX_QUANTIZE_H
#define BFPMX_QUANTIZE_H

#include "arch/prelude.h"

template <
    template <
        std::size_t, BlockDimsType, IFloatRepr
    > typename QuantizationPolicy,
    typename Scalar,
    typename BlockShape, // Concept can not be constrained
    typename Float,
    template <typename> typename ArithmeticPolicy
>
concept IQuantize =
    IFloatRepr<Float> && BlockDimsType<BlockShape> &&
    requires(std::array<f64, BlockShape::TotalSize()> &v,
             Block<Scalar, BlockShape, Float, ArithmeticPolicy, QuantizationPolicy> &b) {
      {
        QuantizationPolicy<sizeof(Scalar), BlockShape, Float>::QuantizerScaleFactor(v)
      } -> std::convertible_to<f64>;

      {
        QuantizationPolicy<sizeof(Scalar), BlockShape, Float>::Identity()
      } -> std::convertible_to<std::string>;
    };

#endif // BFPMX_QUANTIZE_H/