//
// Created by Benjamin White on 25/11/2025.
//

#ifndef BFPMX_JACOBI_H
#define BFPMX_JACOBI_H

#define PROFILE 1
#include "profiler/profiler.h"

#include "prelude.h"

using TestingScalar = u32;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingBlock = Block<TestingScalar, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;

// Somewhat opinionated port of Jacobi2D from PolyBench:
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/3e872547cef7e5c9909422ef1e6af03cf4e56072/stencils/jacobi-2d/jacobi-2d.c
template <size_t N>
static void jacobi_2d_array(
  const int steps,
  std::array<std::array<f64, N>, N> &A,
  std::array<std::array<f64, N>, N> &B
) {
  profiler::func();
  int t, i, j;

  for (t = 0; t < steps; t++)
  {
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        B[i][j] = 0.2f * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        A[i][j] = 0.2f * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
  }
}

template <size_t N>
static void jacobi_2d_block(
  const int steps,
  TestingBlock<BlockDims<N, N>> &A,
  TestingBlock<BlockDims<N, N>> &B
) {
  profiler::func();

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

template <size_t N>
static void jacobi_2d_block_wo_marsh(
  const int steps,
  TestingBlock<BlockDims<N, N>> &A,
  TestingBlock<BlockDims<N, N>> &B
) {
  profiler::func();

  using Dimensions = BlockDims<N, N>;

  // TODO: block mulAt by scalar
  const auto scalar = TestingBlock<BlockDims<1>>(std::array<f64, 1>{0.2f});

  using TypeA = std::remove_reference_t<decltype(A)>;
  using TypeS = decltype(scalar);

  #define AT(i,j)  ((i)*N + (j))
  #define ADD(...) CPUArithmeticSingularValues<TypeA, TypeA, TypeA>::AddAt(__VA_ARGS__);
  #define MUL(...) CPUArithmeticSingularValues<TypeA, TypeA, TypeS>::MulAt(__VA_ARGS__);

  for (u32 t = 0; t < steps; t++) {
    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        ADD(B, AT(i,j), A, AT(i,j), A, AT(i,j-1));
        ADD(B, AT(i,j), B, AT(i,j), A, AT(i,j+1));
        ADD(B, AT(i,j), B, AT(i,j), A, AT(i+1,j));
        ADD(B, AT(i,j), B, AT(i,j), A, AT(i-1,j));
        MUL(B, AT(i,j), B, AT(i,j), scalar, 0);
      }
    }

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        ADD(A, AT(i,j), B, AT(i,j), B, AT(i,j-1));
        ADD(A, AT(i,j), A, AT(i,j), B, AT(i,j+1));
        ADD(A, AT(i,j), A, AT(i,j), B, AT(i+1,j));
        ADD(A, AT(i,j), A, AT(i,j), B, AT(i-1,j));
        MUL(A, AT(i,j), A, AT(i,j), scalar, 0);
      }
    }
  }
}

constexpr u32 N = 128;

int main(int argc, char **argv) {
  using Size = BlockDims<N, N>;
  using Block = TestingBlock<Size>;

  // TODO: unsafe
  double x1 = argc > 1 ? std::stod(argv[1]): 1.2f;
  double x2 = argc > 2 ? std::stod(argv[1]): 3.4f;

  auto a = std::array<std::array<f64, N>, N>{
    std::array<f64, N>{x1},
    std::array<f64, N>{x2},
  };

  auto b = std::array<std::array<f64, N>, N>{
    std::array<f64, N>{x1},
    std::array<f64, N>{x2},
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

  auto dummy_sum_double = [&] (auto &a, auto &b) -> double {
      double sum = 0;
      for (auto &x: a) for (auto &y: x) sum += y;
      for (auto &x: b) for (auto &y: x) sum += y;
      return sum;
  };

  auto dummy_sum_block = [&] (auto &a, auto &b) -> double {
      double sum = 0;
    for (u32 i = 1; i < N - 1; i++) 
      for (u32 j = 1; j < N - 1; j++) 
          sum += a[i, j] + b[i, j];
    return sum;
  };

  // TODO: check results instead of using `dummy_sum`
  int steps = 250;
  profiler::begin();
  jacobi_2d_array<N>(steps, a, b);
  std::cout << dummy_sum_double(a, b) << " ";
  auto blockA = Block(aLinear), blockB = Block(bLinear);
  jacobi_2d_block<N>(steps, blockA, blockB);
  std::cout << dummy_sum_block(blockA, blockB) << " ";
  blockA = Block(aLinear), blockB = Block(bLinear);
  jacobi_2d_block_wo_marsh<N>(steps, blockA, blockB);
  std::cout << dummy_sum_block(blockA, blockB) << "\n";
  profiler::end_and_print();

  return 0;
}

#endif // BFPMX_JACOBI_H
