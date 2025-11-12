#ifndef BFPMX_L2NQ_H
#define BFPMX_L2NQ_H

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"

#include <cstring>

template <std::size_t ScalarBytes, BlockDimsType BlockShape, IFloatRepr Float>
class L2NormQuantization {
public:
  using BlockFmt =
      Block<ScalarBytes, BlockShape, Float, CPUArithmetic, L2NormQuantization>;
  using PackedFloat = std::array<u8, Float::SizeBytes()>;

  static BlockFmt Quantize(std::array<f64, BlockShape::TotalSize()> &vec) {
    f64 sum_of_squares = 0;
    for (int i = 0; i < BlockShape::TotalSize(); i++) {
      sum_of_squares += std::pow(vec[i], 2);
    }

    const f64 scaleFactorFloat = sqrt(sum_of_squares);
    const u32 scaleFactor = lround(log2(scaleFactorFloat));

    std::array<PackedFloat, BlockShape::TotalSize()> blockScaledFloats;
    for (int i = 0; i < BlockShape::TotalSize(); i++) {
      f64 scaledValue = vec[i] / scaleFactor;
      PackedFloat packed = Float::Marshal(scaledValue);
      blockScaledFloats[i] = packed;
    }

    std::array<u8, ScalarBytes> packedScalar;
    for (int i = 0; i < ScalarBytes; i++) {
      packedScalar[i] = static_cast<u8>(scaleFactor >> (i * 8));
    }

    return BlockFmt(blockScaledFloats, packedScalar);
  }
};

#endif // BFPMX_L2NQ_H