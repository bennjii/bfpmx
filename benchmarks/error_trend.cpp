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

constexpr std::string StringOfOperation(const OperationType opera) {
  switch (opera) {
  case AddOp:
    return "Add";
  case SubOp:
    return "Sub";
  case MulOp:
    return "Mul";
  case DivOp:
    return "Div";
  }

  std::cerr << "Invalid /Unknown operation type: " << opera << std::endl;
  exit(1);
}

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
std::vector<NormalVector> GetResults_Nominal(const NormalVector &startingArray,
                                             const OperationType operation) {
  std::vector<NormalVector> results;
  results.reserve(Iterations);
  auto iterationArray = startingArray;

  for (int i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < N; j++) {
      switch (operation) {
      case AddOp: {
        iterationArray[j] += ReferenceArray[j];
      } break;
      case SubOp: {
        iterationArray[j] -= ReferenceArray[j];
      } break;
      case MulOp: {
        iterationArray[j] *= ReferenceArray[j];
      } break;
      case DivOp: {
        iterationArray[j] /= ReferenceArray[j];
      } break;
      }
    }
    results.push_back(iterationArray);
  }
  return results;
}

// Scenario 2: In-block computation. State is kept as a block.
template <IFloatRepr Float, template <std::size_t, BlockDimsType,
                                      IFloatRepr> typename QuantizationPolicy>
std::vector<NormalVector> GetResults_InBlock(const NormalVector &startingArray,
                                             const OperationType operation) {
  std::vector<NormalVector> results;
  results.reserve(Iterations);
  auto activeBlock = TestingBlockT<Float, QuantizationPolicy>(startingArray);
  const auto referenceBlock =
      TestingBlockT<Float, QuantizationPolicy>(ReferenceArray);

  for (int i = 0; i < Iterations; i++) {
    switch (operation) {
    case AddOp: {
      activeBlock = activeBlock + referenceBlock;
    } break;
    case SubOp: {
      activeBlock = activeBlock - referenceBlock;
    } break;
    case MulOp: {
      activeBlock = activeBlock * referenceBlock;
    } break;
    case DivOp: {
      activeBlock = activeBlock / referenceBlock;
    } break;
    }

    results.push_back(activeBlock.Spread());
  }

  return results;
}

constexpr f64 ReferenceValue = 0;
const auto StartingArray = fill_known_arrays<f64, N>(ReferenceValue);

template <IFloatRepr Float, template <std::size_t, BlockDimsType,
                                      IFloatRepr> typename QuantizationPolicy>
void YieldTrend(CsvWriter &writer, OperationType operation) {
  const auto baseline = GetResults_Nominal(StartingArray, operation);
  const std::string label = Float::Nomenclature();

  const CsvInfo block =
      PrepareCsvBlock<TestingBlockT<Float, QuantizationPolicy>>(label, N,
                                                                Iterations);
  const auto results =
      GetResults_InBlock<Float, QuantizationPolicy>(StartingArray, operation);

  for (int i = 0; i < Iterations; i += 1) {
    const f64 percentage = MeanAbsPercentageError(baseline[i], results[i]);
    const f64 absolute = MeanAbsError(baseline[i], results[i]);

    writer.write_err_only(block, StringOfOperation(operation), i, percentage,
                          absolute);
  }
}

template <template <std::size_t, BlockDimsType,
                    IFloatRepr> typename QuantizationPolicy>
void TestQuantizationPolicy(CsvWriter &writer, const OperationType operation) {
  YieldTrend<fp32::E8M23Type, QuantizationPolicy>(writer, operation);
  YieldTrend<fp16::E5M10Type, QuantizationPolicy>(writer, operation);
  YieldTrend<fp8::E4M3Type, QuantizationPolicy>(writer, operation);
  YieldTrend<fp6::E2M3Type, QuantizationPolicy>(writer, operation);
  YieldTrend<fp4::E2M1Type, QuantizationPolicy>(writer, operation);
}

void TestVariants(CsvWriter &writer) {
  for (constexpr std::array operations = {AddOp, SubOp, MulOp, DivOp};
       const auto operation : operations) {
    TestQuantizationPolicy<L2NormQuantization>(writer, operation);
    TestQuantizationPolicy<SharedExponentQuantization>(writer, operation);
    TestQuantizationPolicy<MaximumFractionalQuantization>(writer, operation);
  }
}

int main() {
  auto writer = CsvWriter();
  TestVariants(writer);

  writer.dump("error_trend.csv");
  return 0;
}

#endif // BFPMX_ERROR_TREND_H
