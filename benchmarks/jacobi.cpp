//
// Created by Benjamin White on 25/11/2025.
//

#ifndef BFPMX_JACOBI_H
#define BFPMX_JACOBI_H

#include "prelude.h"

#define PROFILE 1
#include "profiler/csv_info.h"
#include "profiler/profiler.h"

constexpr u32 N = 32;
constexpr u32 Steps = 250;
constexpr u32 Iterations = 100;

using TestingScalar = u32;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions, template <typename> class ArithmeticPolicy>
using TestingBlockT = Block<TestingScalar, Dimensions, TestingFloat,
                            ArithmeticPolicy, SharedExponentQuantization>;

template <typename Dimensions>
using TestingBlock = TestingBlockT<Dimensions, CPUArithmetic>;

template <typename Dimensions>
using TestingBlockNoMarshal =
    TestingBlockT<Dimensions, CPUArithmeticFastMarshalling>;

// Somewhat opinionated port of Jacobi2D from PolyBench:
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/3e872547cef7e5c9909422ef1e6af03cf4e56072/stencils/jacobi-2d/jacobi-2d.c
template <size_t N>
static void Jacobi2DArray(const int steps, std::array<std::array<f64, N>, N> &A,
                          std::array<std::array<f64, N>, N> &B) {
  profiler::func();
  int t, i, j;

  for (t = 0; t < steps; t++) {
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        B[i][j] = 0.2f * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] +
                          A[i - 1][j]);
    for (i = 1; i < N - 1; i++)
      for (j = 1; j < N - 1; j++)
        A[i][j] = 0.2f * (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] +
                          B[i - 1][j]);
  }
}

template <size_t N>
static void Jacobi2DNaiveBlock(const int steps,
                               TestingBlock<BlockDims<N, N>> &A,
                               TestingBlock<BlockDims<N, N>> &B) {
  profiler::func();

  using Dimensions = BlockDims<N, N>;

  for (u32 t = 0; t < steps; t++) {
    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        u32 coords = Dimensions::CoordsToLinear({i, j});
        f64 newVal = 0.2f * (A[i, j] + A[i, j - 1] + A[i, 1 + j] + A[1 + i, j] +
                             A[i - 1, j]);

        B.SetValue(coords, newVal);
      }
    }

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        u32 coords = Dimensions::CoordsToLinear({i, j});

        f64 newVal = 0.2f * (B[i, j] + B[i, j - 1] + B[i, 1 + j] + B[1 + i, j] +
                             B[i - 1, j]);
        A.SetValue(coords, newVal);
      }
    }
  }
}

template <size_t N>
static void Jacobi2DSpreadBlockEach(const int steps,
                                    TestingBlock<BlockDims<N, N>> &A_block,
                                    TestingBlock<BlockDims<N, N>> &B_block) {
  profiler::func();

  using Dimensions = BlockDims<N, N>;

  std::array<f64, N * N> a_spread, b_spread;

  for (u32 t = 0; t < steps; t++) {
    a_spread = A_block.Spread();
    std::array<f64, N * N> b_new_values = a_spread;

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        const u32 center = i * N + j;
        const u32 left = i * N + (j - 1);
        const u32 right = i * N + (j + 1);
        const u32 up = (i - 1) * N + j;
        const u32 down = (i + 1) * N + j;
        b_new_values[center] =
            0.2 * (a_spread[center] + a_spread[left] + a_spread[right] +
                   a_spread[down] + a_spread[up]);
      }
    }
    B_block = TestingBlock<Dimensions>(b_new_values);

    b_spread = B_block.Spread();
    std::array<f64, N * N> a_new_values = b_spread;

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        const u32 center = i * N + j;
        const u32 left = i * N + (j - 1);
        const u32 right = i * N + (j + 1);
        const u32 up = (i - 1) * N + j;
        const u32 down = (i + 1) * N + j;
        a_new_values[center] =
            0.2 * (b_spread[center] + b_spread[left] + b_spread[right] +
                   b_spread[down] + b_spread[up]);
      }
    }
    A_block = TestingBlock<Dimensions>(a_new_values);
  }
}

template <size_t N>
static void Jacobi2DSpreadBlockOnce(const int steps,
                                    TestingBlock<BlockDims<N, N>> &A_block,
                                    TestingBlock<BlockDims<N, N>> &B_block) {
  profiler::func();

  using Dimensions = BlockDims<N, N>;

  std::array<f64, N * N> a_spread, b_spread;
  a_spread = A_block.Spread();
  b_spread = B_block.Spread();

  for (u32 t = 0; t < steps; t++) {
    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        const u32 center = i * N + j;
        const u32 left = i * N + (j - 1);
        const u32 right = i * N + (j + 1);
        const u32 up = (i - 1) * N + j;
        const u32 down = (i + 1) * N + j;
        b_spread[center] =
            0.2 * (a_spread[center] + a_spread[left] + a_spread[right] +
                   a_spread[down] + a_spread[up]);
      }
    }

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        const u32 center = i * N + j;
        const u32 left = i * N + (j - 1);
        const u32 right = i * N + (j + 1);
        const u32 up = (i - 1) * N + j;
        const u32 down = (i + 1) * N + j;
        a_spread[center] =
            0.2 * (b_spread[center] + b_spread[left] + b_spread[right] +
                   b_spread[down] + b_spread[up]);
      }
    }
  }

  A_block = TestingBlock<Dimensions>(a_spread);
  B_block = TestingBlock<Dimensions>(b_spread);
}

template <size_t N>
static void Jacobi2DAlwaysFastMarshal(const int steps,
                                      TestingBlock<BlockDims<N, N>> &A_block,
                                      TestingBlock<BlockDims<N, N>> &B_block) {
  profiler::func();
  using Dimensions = BlockDims<N, N>;

  // TODO: block mulAt by scalar
  const auto scalar = TestingBlock<BlockDims<1>>(std::array<f64, 1>{0.2f});

  using TypeA = std::remove_reference_t<decltype(A_block)>;
  using TypeS = decltype(scalar);

#define AT(i, j) ((i) * N + (j))
#define ADD(...)                                                               \
  CPUArithmeticSingularValues<TypeA, TypeA, TypeA>::AddAt(__VA_ARGS__);
#define MUL(...)                                                               \
  CPUArithmeticSingularValues<TypeA, TypeA, TypeS>::MulAt(__VA_ARGS__);

  for (u32 t = 0; t < steps; t++) {
    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        ADD(B_block, AT(i, j), A_block, AT(i, j), A_block, AT(i, j - 1));
        ADD(B_block, AT(i, j), B_block, AT(i, j), A_block, AT(i, j + 1));
        ADD(B_block, AT(i, j), B_block, AT(i, j), A_block, AT(i + 1, j));
        ADD(B_block, AT(i, j), B_block, AT(i, j), A_block, AT(i - 1, j));
        MUL(B_block, AT(i, j), B_block, AT(i, j), scalar, 0);
      }
    }

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        ADD(A_block, AT(i, j), B_block, AT(i, j), B_block, AT(i, j - 1));
        ADD(A_block, AT(i, j), A_block, AT(i, j), B_block, AT(i, j + 1));
        ADD(A_block, AT(i, j), A_block, AT(i, j), B_block, AT(i + 1, j));
        ADD(A_block, AT(i, j), A_block, AT(i, j), B_block, AT(i - 1, j));
        MUL(A_block, AT(i, j), A_block, AT(i, j), scalar, 0);
      }
    }
  }
}

void Test() {
  using Size = BlockDims<N, N>;
  using Block = TestingBlock<Size>;

  auto a = std::array<std::array<f64, N>, N>{
      std::array<f64, N>{1.2f},
      std::array<f64, N>{3.4f},
  };

  auto b = std::array<std::array<f64, N>, N>{
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

  // used to prevent compiler optimization on the calls
  auto black_box_f64 = [&](auto &a, auto &b) {
    f64 sum = 0;
    for (u32 i = 0; i < N; i++)
      for (u32 j = 0; j < N; j++)
        sum += a[i][j] + b[i][j];
    volatile f64 x = sum;
  };

  // used to prevent compiler optimization on the calls
  auto black_box_block = [&](auto &a, auto &b) {
    f64 sum = 0;
    for (u32 i = 0; i < N; i++)
      for (u32 j = 0; j < N; j++)
        sum += a[i, j] + b[i, j];
    volatile f64 x = sum;
  };

  Jacobi2DArray<N>(Steps, a, b);
  black_box_f64(a, b);

  Block blockA, blockB;

  blockA = Block(aLinear);
  blockB = Block(bLinear);
  Jacobi2DNaiveBlock<N>(Steps, blockA, blockB);
  black_box_block(blockA, blockB);

  blockA = Block(aLinear);
  blockB = Block(bLinear);
  Jacobi2DSpreadBlockEach<N>(Steps, blockA, blockB);
  black_box_block(blockA, blockB);

  blockA = Block(aLinear);
  blockB = Block(bLinear);
  Jacobi2DSpreadBlockOnce<N>(Steps, blockA, blockB);
  black_box_block(blockA, blockB);

  blockA = Block(aLinear);
  blockB = Block(bLinear);
  Jacobi2DAlwaysFastMarshal<N>(Steps, blockA, blockB);
  black_box_block(blockA, blockB);
}

int main() {
  using Size = BlockDims<N, N>;
  using Block = TestingBlock<Size>;

  CsvInfo primitive = PrepareCsvPrimitive("jacobi2d:primitive", N, Steps);
  CsvInfo block = PrepareCsvBlock<Block>("jacobi2d:block", N, Steps);

  auto to_dump = CsvWriter();

  profiler::begin();

  for (int i = 0; i < Iterations; i++) {
    Test();

    to_dump.next_iteration();
    auto infos = profiler::dump_and_reset();

    for (auto &x : infos) {
      auto &y = (std::string(x.label) == "Jacobi2DArray" ? primitive : block);
      to_dump.append_csv(y, x, 0);
    }
  }

  to_dump.dump("jacobi2d.csv");
  // to_dump.dump(std::cout);

  return 0;
}

#endif // BFPMX_JACOBI_H
