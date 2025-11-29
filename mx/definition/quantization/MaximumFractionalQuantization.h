//
// Created by Benjamin White on 23/10/2025.
//

#ifndef BFPMX_MFQ_H
#define BFPMX_MFQ_H

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"

#include <iostream>

template <std::size_t ScalarBytes, BlockDimsType BlockShape, IFloatRepr Float>
class MaximumFractionalQuantization {
public:
  static f64
  QuantizerScaleFactor(const std::array<f64, BlockShape::TotalSize()> &vec) {
    f64 largestValue = 0;
    for (int i = 0; i < BlockShape::TotalSize(); i++) {
      const f64 absValue = fabs(vec[i]);
      largestValue = std::max(largestValue, absValue);
    }

    const u32 scaleFactorCandidate = lround(largestValue);
    const u32 scaleFactorInt = 31 - __builtin_clz(scaleFactorCandidate);
    const u32 scaleFactor = 1 << scaleFactorInt;

    return scaleFactor;
  }

  static std::string Identity() { return "MaximumFractionalQuantization"; }

private:
  static f64 ScaleFactor(f64 HighestValueAbsolute) {
    return HighestValueAbsolute / Float::BiasValue();
  }
};

#endif // BFPMX_MFQ_H