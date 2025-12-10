//
// Created by Benjamin White on 25/11/2025.
//

#ifndef BFPMX_JACOBI_H
#define BFPMX_JACOBI_H

#define PROFILE 1

#include "prelude.h"

#include "profiler/csv_info.h"
#include "profiler/profiler.h"

constexpr u32 N = 32;
constexpr std::array<u32,4> StepsArray = {5,10,50,100};
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
        u32 coords = BlockDims<N, N>::CoordsToLinear({i, j});
        f64 newVal = 0.2f * (A[i, j] + A[i, j - 1] + A[i, 1 + j] + A[1 + i, j] +
                             A[i - 1, j]);

        B.SetValue(coords, newVal);
      }
    }

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        u32 coords = BlockDims<N, N>::CoordsToLinear({i, j});

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

  A_block = TestingBlock<BlockDims<N, N>>(a_spread);
  B_block = TestingBlock<BlockDims<N, N>>(b_spread);
}

template <size_t N_>
static f64 L2Norm(const std::array<std::array<f64, N_>, N_> &A,
                  const std::array<f64, N_ * N_> &B_linear) {
  f64 error = 0.0;
  for (size_t i = 0; i < N_; ++i) {
    for (size_t j = 0; j < N_; ++j) {
      f64 diff = A[i][j] - B_linear[i * N_ + j];
      error += diff * diff;
    }
  }
  return std::sqrt(error);
}

template <size_t N_>
static f64 L2Norm(const std::array<std::array<f64, N_>, N_> &A) {
  f64 norm_sq = 0.0;
  for (size_t i = 0; i < N_; ++i) {
    for (size_t j = 0; j < N_; ++j) {
      norm_sq += A[i][j] * A[i][j];
    }
  }
  return std::sqrt(norm_sq);
}

struct ElementWise {
  f64 naive;
  f64 spread_each;
  f64 spread_once;
};

struct Iteration {
  ElementWise percentage;
  ElementWise absolute;
};

Iteration Test(u32 Steps) {
  using Size = BlockDims<N, N>;
  using Block = TestingBlock<Size>;

  // Create and fill arrays
  auto a_base = std::array<std::array<f64, N>, N>{};
  auto b_base = std::array<std::array<f64, N>, N>{};

  for (u32 i = 0; i < N; i++) {
    a_base[i] = fill_random_arrays<f64, N>(-10, 10);
    b_base[i] = fill_random_arrays<f64, N>(-10, 10);
  }

  // Linearize for blocks
  std::array<f64, N * N> aLinear_base = {};
  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      aLinear_base[i * N + j] = a_base[i][j];
    }
  }

  std::array<f64, N * N> bLinear_base = {};
  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      bLinear_base[i * N + j] = b_base[i][j];
    }
  }

  // Get reference result
  auto a_ref = a_base;
  auto b_ref = b_base;
  Jacobi2DArray<N>(Steps, a_ref, b_ref);

  profiler::begin();

  auto a_prof = a_base;
  auto b_prof = b_base;
  Jacobi2DArray<N>(Steps, a_prof, b_prof);

  Block blockA_naive(aLinear_base), blockB_naive(bLinear_base);
  Jacobi2DNaiveBlock<N>(Steps, blockA_naive, blockB_naive);

  Block blockA_spread_each(aLinear_base), blockB_spread_each(bLinear_base);
  Jacobi2DSpreadBlockEach<N>(Steps, blockA_spread_each, blockB_spread_each);

  Block blockA_spread_once(aLinear_base), blockB_spread_once(bLinear_base);
  Jacobi2DSpreadBlockOnce<N>(Steps, blockA_spread_once, blockB_spread_once);

  auto norm_ref = L2Norm(a_base);

  const auto collect_error_percent = [&](const f64 error_abs) {
    return (error_abs / norm_ref) * 100.0;
  };

  const auto error_naive = L2Norm<N>(a_ref, blockA_naive.Spread());
  const auto error_spread_each = L2Norm<N>(a_ref, blockA_spread_each.Spread());
  const auto error_spread_once = L2Norm<N>(a_ref, blockA_spread_once.Spread());

  return Iteration{
      ElementWise{error_naive, error_spread_each, error_spread_once},
      ElementWise{collect_error_percent(error_naive),
                  collect_error_percent(error_spread_each),
                  collect_error_percent(error_spread_once)},
  };
}

int main() {
  using Size = BlockDims<N, N>;
  using Block = TestingBlock<Size>;
  auto writer = CsvWriter();
  for (u32 Steps: StepsArray){
    CsvInfo primitive = PrepareCsvPrimitive("jacobi2d:primitive", N, Steps);
    CsvInfo block = PrepareCsvBlock<Block>("jacobi2d:block", N, Steps);


    profiler::begin();

    for (int i = 0; i < Iterations; i++) {
      auto [percentage, absolute] = Test(Steps);

      writer.next_iteration();
      auto infos = profiler::dump_and_reset();

      for (auto &x : infos) {
        auto const &label = std::string(x.label);

        if (label == "Jacobi2DArray") {
          writer.append_csv(primitive, x, 0, 0);
        } else if (label == "Jacobi2DNaiveBlock") {
          writer.append_csv(block, x, percentage.naive, absolute.naive);
        } else if (label == "Jacobi2DSpreadBlockEach") {
          writer.append_csv(block, x, percentage.spread_each,
                            absolute.spread_each);
        } else if (label == "Jacobi2DSpreadBlockOnce") {
          writer.append_csv(block, x, percentage.spread_once,
                            absolute.spread_once);
        }
      }
    }
  }

  writer.dump("../benchmarks/jacobi2d.csv");
  // to_dump.dump(std::cout);

  return 0;
}

#endif // BFPMX_JACOBI_H
