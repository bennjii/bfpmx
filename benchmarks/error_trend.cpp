//
// Created by Benjamin White on 2/12/2025.
//

#ifndef BFPMX_ERROR_TREND_H
#define BFPMX_ERROR_TREND_H

#define PROFILE 1

#include "definition/block_float/block/BlockDims.h"
#include "prelude.h"
#include "profiler/csv_info.h"
#include "profiler/profiler.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr u64 Iterations = 250;
constexpr u64 N = 32;

using TestingScalar = u32;

template <IFloatRepr Float, template <std::size_t, BlockDimsType,
              IFloatRepr> typename QuantizationPolicy>
using TestingBlockT = Block<TestingScalar, BlockDims<N>, Float, CPUArithmetic,
                            QuantizationPolicy>;

using NormalVector = std::array<f64, N>;

const std::array<f64, N> ReferenceArray = fill_known_arrays<f64, N>(2.0);

f64 MeanAbsError(const NormalVector &A, const NormalVector &B) {
  f64 sumAbs = 0.0;
  for (size_t i = 0; i < N; i++) {
    sumAbs += std::abs(A[i] - B[i]);
  }
  return sumAbs / N;
}

f64 MeanAbsPercentageError(const NormalVector &A, const NormalVector &B) {
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
template <IFloatRepr Float, template <std::size_t, BlockDimsType,
              IFloatRepr> typename QuantizationPolicy>
std::vector<NormalVector>
GetResults_InBlock(const NormalVector &startingArray) {
  std::vector<NormalVector> results;
  results.reserve(Iterations);
  auto activeBlock = TestingBlockT<Float, QuantizationPolicy>(startingArray);
  const auto referenceBlock = TestingBlockT<Float, QuantizationPolicy>(ReferenceArray);

  for (int i = 0; i < Iterations; i++) {
    activeBlock = activeBlock + referenceBlock;
    results.push_back(activeBlock.Spread());
  }

  return results;
}

constexpr f64 ReferenceValue = 0;
const auto StartingArray = fill_known_arrays<f64, N>(ReferenceValue);

template <IFloatRepr Float, template <std::size_t, BlockDimsType,
              IFloatRepr> typename QuantizationPolicy>
void YieldTrend(CsvWriter& writer) {
  const auto baseline = GetResults_Nominal(StartingArray);
  const std::string label = Float::Nomenclature();

  const CsvInfo block = PrepareCsvBlock<TestingBlockT<Float, QuantizationPolicy>>(label, N, Iterations);
  const auto results = GetResults_InBlock<Float, QuantizationPolicy>(StartingArray);

  for (int i = 0; i < Iterations; i += 1) {
    const f64 percentage = MeanAbsPercentageError(baseline[i], results[i]);
    const f64 absolute = MeanAbsError(baseline[i], results[i]);

    writer.write_err_only(block, label, i, percentage, absolute);
  }
}

template <template <std::size_t, BlockDimsType,
              IFloatRepr> typename QuantizationPolicy>
void TestQuantizationPolicy(CsvWriter& writer) {
  YieldTrend<fp64::E11M52Type, QuantizationPolicy>(writer);
  YieldTrend<fp32::E8M23Type, QuantizationPolicy>(writer);
  YieldTrend<fp16::E5M10Type, QuantizationPolicy>(writer);
  YieldTrend<fp8::E4M3Type, QuantizationPolicy>(writer);
  YieldTrend<fp6::E2M3Type, QuantizationPolicy>(writer);
  YieldTrend<fp4::E2M1Type, QuantizationPolicy>(writer);
}

int main() {
  auto writer = CsvWriter();

  TestQuantizationPolicy<L2NormQuantization>(writer);
  TestQuantizationPolicy<SharedExponentQuantization>(writer);
  TestQuantizationPolicy<MaximumFractionalQuantization>(writer);

  writer.dump("error_trend.csv");
  return 0;
}

#endif // BFPMX_ERROR_TREND_H