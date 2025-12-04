//
// Created by Benjamin White on 2/12/2025.
//

#ifndef BFPMX_ERROR_TREND_H
#define BFPMX_ERROR_TREND_H

#define PROFILE 1

#include "prelude.h"
#include "profiler/profiler.h"
#include "definition/block_float/block/BlockDims.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr u32 Iterations = 25;
constexpr u32 N = 32;

using TestingScalar = u32;

template <IFloatRepr Float>
using TestingBlockT = Block<TestingScalar, BlockDims<N>, Float,
                            CPUArithmetic, L2NormQuantization>;

using NormalVector = std::array<f64, N>;

const std::array<f64, N> ReferenceArray = fill_known_arrays<f64, N>(2.0);

f64 MeanAbsError(const NormalVector& A, const NormalVector& B) {
  f64 sumAbs = 0.0;
  for (size_t i = 0; i < N; i++) {
    sumAbs += std::abs(A[i] - B[i]);
  }
  return sumAbs / N;
}

f64 MeanAbsPercentageError(const NormalVector& A, const NormalVector& B) {
    f64 sumPerc = 0.0;
    f64 nonZeroCount = 0;
    for (size_t i = 0; i < N; i++) {
        if (A[i] != 0.0) {
            sumPerc += std::abs((A[i] - B[i]) / A[i]);
            nonZeroCount++;
        }
    }
    if (nonZeroCount == 0) {
        return 0.0;
    }
    return (sumPerc / nonZeroCount) * 100.0;
}

// Scenario 1: State is kept as f64.
std::vector<NormalVector>
GetResults_Nominal(const NormalVector &startingArray) {
    std::vector<NormalVector> results;
    results.reserve(Iterations);
    auto iterationArray = startingArray;

    for (int i = 0; i < Iterations; i++) {
        for (size_t j = 0; j < N; j++) {
            iterationArray[j] += ReferenceArray[j];
        }
        results.push_back(iterationArray);
    }
    return results;
}

// Scenario 2: In-block computation. State is kept as a block.
template<IFloatRepr Float>
std::vector<NormalVector>
GetResults_InBlock(const NormalVector &startingArray) {
  std::vector<NormalVector> results;
  results.reserve(Iterations);
  auto activeBlock = TestingBlockT<Float>(startingArray);
  const auto referenceBlock = TestingBlockT<Float>(ReferenceArray);

  for (int i = 0; i < Iterations; i++) {
    activeBlock = activeBlock + referenceBlock;
    results.push_back(activeBlock.Spread());
  }

  return results;
}

int main() {
  const auto startingArray = fill_known_arrays<f64, N>(0);

  auto requant_results = GetResults_Nominal(startingArray);

  auto inblock_results_fp16 = GetResults_InBlock<fp16::E5M10Type>(startingArray);
  auto inblock_results_fp08 = GetResults_InBlock<fp8::E4M3Type>(startingArray);
  auto inblock_results_fp06 = GetResults_InBlock<fp6::E2M3Type>(startingArray);
  auto inblock_results_fp04 = GetResults_InBlock<fp4::E2M1Type>(startingArray);

  // --- Error Analysis ---
  std::cout << std::fixed << std::setprecision(4);

  std::cout << "\n--- Divergence from Re-quantized Strategy (Mean Absolute Percentage Error) ---" << std::endl;
  std::cout << "Measures deviation from the re-quantized baseline to isolate arithmetic error sources." << std::endl;
  std::cout << "Iter\tFP16 (%)\t\t\tFP8 (%)\t\t\t\tFP6 (%)\t\t\t\tFP4 (%)" << std::endl;
  for (int i = 0; i < Iterations; i += 1) {
      const auto& baseline = requant_results[i];

      const f64 error1 = MeanAbsPercentageError(baseline, inblock_results_fp16[i]);
      const f64 error1_abs = MeanAbsError(baseline, inblock_results_fp16[i]);

      const f64 error2 = MeanAbsPercentageError(baseline, inblock_results_fp08[i]);
      const f64 error2_abs = MeanAbsError(baseline, inblock_results_fp08[i]);

      const f64 error3 = MeanAbsPercentageError(baseline, inblock_results_fp06[i]);
      const f64 error3_abs = MeanAbsError(baseline, inblock_results_fp06[i]);

      const f64 error4 = MeanAbsPercentageError(baseline, inblock_results_fp04[i]);
      const f64 error4_abs = MeanAbsError(baseline, inblock_results_fp04[i]);

      std::cout << (i + 1) << "\t"
        << error1 << " (" << "error=" << error1_abs << ")" << "\t\t"
        << error2 << " (" << "error=" << error2_abs << ")" << "\t\t"
        << error3 << " (" << "error=" << error3_abs << ")" << "\t\t"
        << error4 << " (" << "error=" << error4_abs << ")"
        << std::endl;
  }

  return 0;
}

#endif // BFPMX_ERROR_TREND_H