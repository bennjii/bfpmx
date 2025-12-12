#ifndef BFPMX_SEQ_H
#define BFPMX_SEQ_H

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"

#include <cstring>

template <std::size_t ScalarBytes, BlockDimsType BlockShape, IFloatRepr Float>
class SharedExponentQuantization {
public:
  static f64
  QuantizerScaleFactor(const std::array<f64, BlockShape::TotalSize()> &vec) {
    uint64_t largestBiasedExponent = 0;
    for (int i = 0; i < BlockShape::TotalSize(); i++) {
      const uint64_t bits = std::bit_cast<uint64_t>(vec[i]);
      const uint64_t exponent = (bits >> F64_BITS_SIGNIFICAND) & 0x7FF;
      largestBiasedExponent = std::max(largestBiasedExponent, exponent);
    }

    const u64 largestUnbiasedExponent = (largestBiasedExponent > F64_BIAS)
                                            ? largestBiasedExponent - F64_BIAS
                                            : 0;
    const u64 exponent = NormalizedExponent(largestUnbiasedExponent);

    return std::pow(2.0, exponent);
  }

  static std::string Identity() { return "SharedExponentQuantization"; }

private:
  static constexpr u64 MaximumScalarExponentValue() {
    return (static_cast<u64>(1) << (ScalarBytes * 8)) - 1;
  }

  static u64 NormalizedExponent(const u64 highestExponent) {
    // Exponent cannot exceed the representable maximum.
    return std::min(MaximumScalarExponentValue(), highestExponent);
  }
};

#endif // BFPMX_SEQ_H