//
// Created by Benjamin White on 25/11/2025.
//

#ifndef BFPMX_JACOBI_H
#define BFPMX_JACOBI_H

#define PROFILE 1

#include "prelude.h"
#include "profiler/profiler.h"

constexpr u32 TestingScalarSize = 4;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingBlock = Block<TestingScalarSize, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;

// Somewhat opinionated port of Jacobi2D from PolyBench:
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/3e872547cef7e5c9909422ef1e6af03cf4e56072/stencils/jacobi-2d/jacobi-2d.c
template <size_t N>
static void jacobi_2d_array(
  const int steps,
  std::array<std::array<f64, N>, N> A,
  std::array<std::array<f64, N>, N> B
) {
  profiler::function();
  int t, i, j;

#pragma scop
  for (t = 0; t < steps; t++)
  {
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        B[i][j] = 0.2f * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        A[i][j] = 0.2f * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
  }
#pragma endscop
}

template <size_t N>
static void jacobi_2d_block(
  const int steps,
  TestingBlock<BlockDims<N, N>> A,
  TestingBlock<BlockDims<N, N>> B
) {
  profiler::function();

  using Dimensions = BlockDims<N, N>;

  for (u32 t = 0; t < steps; t++) {
    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        u32 coords = BlockDims<N, N>::CoordsToLinear({i, j});
        f64 newVal = 0.2f * (A[i, j] + A[i, j - 1] + A[i, 1 + j] + A[1 + i, j] + A[i - 1, j]);

        B.SetValue(coords, newVal);
      }
    }

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        u32 coords = BlockDims<N, N>::CoordsToLinear({i, j});

        f64 newVal = 0.2f * (B[i, j] + B[i, j - 1] + B[i, 1 + j] + B[1 + i, j] + B[i - 1, j]);
        A.SetValue(coords, newVal);
      }
    }
  }
}

constexpr u32 N = 32;

int main() {
  profiler::begin();

  using Size = BlockDims<N, N>;
  using Block = TestingBlock<Size>;

  const auto a = std::array<std::array<f64, N>, N>{
    std::array<f64, N>{1.2f},
    std::array<f64, N>{3.4f},
  };

  const auto b = std::array<std::array<f64, N>, N>{
    std::array<f64, N>{1.2f},
    std::array<f64, N>{3.4f},
  };

  std::array<f64, N * N> aLinear = {};
  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      aLinear[i * N + j] = a[i][j];
    }
  }

  std::array<f64, N * N> bLinear = {};
  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      bLinear[i * N + j] = b[i][j];
    }
  }

  const auto blockA = Block(aLinear);
  const auto blockB = Block(bLinear);

  jacobi_2d_array<N>(250, a, b);
  jacobi_2d_block<N>(250, blockA, blockB);

  profiler::end_and_print();
  return 0;
}

#endif // BFPMX_JACOBI_H
