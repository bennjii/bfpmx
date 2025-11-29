#ifndef BFPMX_L2NQ_H
#define BFPMX_L2NQ_H

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"

#include <cstring>

template <std::size_t ScalarBytes, BlockDimsType BlockShape, IFloatRepr Float>
class L2NormQuantization {
public:
  static f64
  QuantizerScaleFactor(const std::array<f64, BlockShape::TotalSize()> &vec) {
    f64 sum_of_squares = 0;
    for (int i = 0; i < BlockShape::TotalSize(); i++) {
      sum_of_squares += std::pow(vec[i], 2);
    }

    const f64 scaleFactorFloat = sqrt(sum_of_squares / BlockShape::TotalSize());
    const u32 scaleFactor = lround(scaleFactorFloat);

    return scaleFactor;
  }

  static std::string Identity() { return "L2NormQuantization"; }
};

#endif // BFPMX_L2NQ_H