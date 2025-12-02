//
// Created by Benjamin White on 2/12/2025.
//

#ifndef BFPMX_JACOBI_H
#define BFPMX_JACOBI_H

#define PROFILE 1

#include "prelude.h"
#include "profiler/profiler.h"

constexpr u32 Iterations = 100;
constexpr u32 N = 32;

using TestingScalar = u32;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions, template <typename> class ArithmeticPolicy>
using TestingBlockT = Block<TestingScalar, Dimensions, TestingFloat,
                            ArithmeticPolicy, SharedExponentQuantization>;

using TestingBlock = TestingBlockT<BlockDims<N>, CPUArithmetic>;
using NormalVector = std::array<f64, N>;

const std::array<f64, N> ReferenceArray = fill_known_arrays<f64, N>(2.0);

f64 MeanAbsError(const NormalVector A, const NormalVector B) {
  f64 sumAbs = 0.0;

  for (size_t i = 0; i < N; i++) {
      sumAbs += std::abs(A[i] - B[i]);
  }

  return sumAbs / N;
}

std::array<f64, N> Iteration(const std::array<f64, N> &array) {
  const auto referenceBlock = TestingBlock(ReferenceArray);
  const auto activeBlock = TestingBlock(array);

  const auto addedBlocks = referenceBlock + activeBlock;
  return addedBlocks.Spread();
}

std::array<f64, Iterations> Test() {
  auto startingArray = fill_random_arrays<f64, N>(-10, 10);
  auto iterationArray = startingArray;

  std::array<f64, Iterations> error{};

  for (int i = 0; i < Iterations; i++) {
    iterationArray = Iteration(iterationArray);
    error[i] = MeanAbsError(startingArray, iterationArray);
  }

  return error;
}

int main() {
  profiler::begin();

  for (int i = 0; i < Iterations; i++) {
    Test();
  }

  profiler::end_and_print();
  return 0;
}

#endif // BFPMX_JACOBI_H