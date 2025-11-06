#ifndef BFPMX_SEQ_H
#define BFPMX_SEQ_H

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"

#include <cstring>
#include <iostream>

template <std::size_t ScalarBytes, BlockDimsType BlockShape, IFloatRepr Float>
class SharedExponentQuantization {
public:
  using BlockFmt = Block<ScalarBytes, BlockShape, Float, CPUArithmetic,
                         SharedExponentQuantization>;
  using PackedFloat = std::array<u8, Float::SizeBytes()>;

  static BlockFmt Quantize(const std::array<f64, BlockShape::TotalSize()> &vec) {
    u64 largestBiasedExponent = 0;
    for (int i = 0; i < BlockShape::TotalSize(); i++) {
      u64 bits;
      std::memcpy(&bits, &vec[i], sizeof(bits));
      const u64 exponent = (bits >> 52) & 0x7FF;

      if (exponent > largestBiasedExponent) {
        largestBiasedExponent = exponent;
      }
    }

    const u64 largestUnbiasedExponent = largestBiasedExponent - 1023;
    const u64 exponent = NormalizedExponent(largestUnbiasedExponent);
    f64 scaleFactor = std::pow(2.0, exponent);

    std::array<PackedFloat, BlockShape::TotalSize()> blockScaledFloats;
    for (int i = 0; i < BlockShape::TotalSize(); i++) {
      f64 scaledValue = vec[i] / scaleFactor;
      PackedFloat packed = Float::Marshal(scaledValue);
      blockScaledFloats[i] = packed;
    }

    const u32 scaleFactorInt = lround(exponent);

    std::array<u8, ScalarBytes> packedScalar;
    for (int i = 0; i < ScalarBytes; i++) {
      packedScalar[i] = static_cast<u8>(scaleFactorInt >> (i * 8));
    }

    return BlockFmt(blockScaledFloats, packedScalar);
  }

  static std::string Identity() {
    return "SharedExponentQuantization";
  }

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