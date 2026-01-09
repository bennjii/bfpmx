//
// Created by Gemini on 12/9/2025.
//

#ifndef BFPMX_ERROR_PURE_H
#define BFPMX_ERROR_PURE_H

#define PROFILE 1

#include "definition/block_float/block/BlockDims.h"
#include "prelude.h"
#include "profiler/csv_info.h"
#include "profiler/profiler.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <array>

constexpr u64 N = 32;

using TestingScalar = u32;

template <IFloatRepr Float, template <std::size_t, BlockDimsType,
                                      IFloatRepr> typename QuantizationPolicy>
using TestingBlockT = Block<TestingScalar, BlockDims<N>, Float, CPUArithmetic,
                            QuantizationPolicy>;

using NormalVector = std::array<f64, N>;

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

template <IFloatRepr Float, template <std::size_t, BlockDimsType,
                                      IFloatRepr> typename QuantizationPolicy>
void YieldPureError(CsvWriter &writer) {
  const std::string label = Float::Nomenclature();
  
  // Create a CsvInfo block. We'll set steps to 0 or total iterations if known.
  // We are iterating exp from -3.0 to 6.0 with step 0.1 -> 90 steps approx.
  const CsvInfo block =
      PrepareCsvBlock<TestingBlockT<Float, QuantizationPolicy>>(label, N, 0);

  // Logarithmic steps: 10^-3 (0.001) to 10^6 (1,000,000)
  const f64 start_exp = -3.0; 
  const f64 end_exp = 6.0;    
  const f64 step_exp = 0.01;

  for (f64 exp = start_exp; exp <= end_exp + 1e-9; exp += step_exp) {
      f64 v = std::pow(10.0, exp);
      
      NormalVector input;
      input.fill(v);
      
      // Initialize block with 'v' (quantization happens here)
      auto testBlock = TestingBlockT<Float, QuantizationPolicy>(input);
      // Get the values back (dequantization happens here)
      NormalVector output = testBlock.Spread();
      
      f64 percentage = MeanAbsPercentageError(input, output);
      f64 absolute = MeanAbsError(input, output);
      
      // We use 'v' as the iteration variable to plot against x-axis (value magnitude)
      writer.write_err_only(block, "PureQuant", v, percentage, absolute);
  }
}

template <template <std::size_t, BlockDimsType,
                    IFloatRepr> typename QuantizationPolicy>
void TestQuantizationPolicy(CsvWriter &writer) {
  YieldPureError<fp32::E8M23Type, QuantizationPolicy>(writer);
  YieldPureError<fp16::E5M10Type, QuantizationPolicy>(writer);
  YieldPureError<fp8::E4M3Type, QuantizationPolicy>(writer);
  YieldPureError<fp6::E3M2Type, QuantizationPolicy>(writer);
  YieldPureError<fp4::E2M1Type, QuantizationPolicy>(writer);
}

void TestVariants(CsvWriter &writer) {
  // TestQuantizationPolicy<L2NormQuantization>(writer);
  // TestQuantizationPolicy<SharedExponentQuantization>(writer);
  TestQuantizationPolicy<MaximumFractionalQuantization>(writer);
}

int main() {
  auto writer = CsvWriter();
  TestVariants(writer);

  writer.dump("error_pure.csv");
  std::cout << "Written file error_pure.csv" << std::endl;
  return 0;
}

#endif // BFPMX_ERROR_PURE_H
