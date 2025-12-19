#define PROFILE 1

#include "prelude.h"

#include "profiler/csv_info.h"
#include "profiler/profiler.h"

constexpr u32 N = 32;
constexpr std::array<u32, 4> StepsArray = {5, 10, 50, 100};
constexpr u32 Iterations = 100;

using TestingScalar = u32;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions, template <typename> class ArithmeticPolicy>
using TestingBlockT = Block<TestingScalar, Dimensions, TestingFloat,
                            ArithmeticPolicy, SharedExponentQuantization>;

template <typename Dimensions>
using TestingBlock = TestingBlockT<Dimensions, CPUArithmetic>;

// Reference implementation based on PolyBench Seidel-2D
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/stencils/seidel-2d/seidel-2d.c
template <size_t N>
static void Seidel2DArray(const int steps, std::array<std::array<f64, N>, N> &A) {
  profiler::func();
  int t, i, j;

  for (t = 0; t < steps; t++) {
    for (i = 1; i < N - 1; i++) {
      for (j = 1; j < N - 1; j++) {
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] +
                   A[i][j - 1] + A[i][j] + A[i][j + 1] +
                   A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) /
                  9.0;
      }
    }
  }
}

template <size_t N>
static void Seidel2DNaiveBlock(const int steps,
                               TestingBlock<BlockDims<N, N>> &A) {
  profiler::func();

  for (u32 t = 0; t < steps; t++) {
    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        u32 coords = BlockDims<N, N>::CoordsToLinear({i, j});
        
        // Read 9 values
        // Note: operator[] accesses the block values (possibly decoding them)
        f64 val = (A[i - 1, j - 1] + A[i - 1, j] + A[i - 1, j + 1] +
                   A[i, j - 1] + A[i, j] + A[i, j + 1] +
                   A[i + 1, j - 1] + A[i + 1, j] + A[i + 1, j + 1]) /
                  9.0;

        A.SetValue(coords, val);
      }
    }
  }
}

template <size_t N>
static void Seidel2DSpreadBlockEach(const int steps,
                                    TestingBlock<BlockDims<N, N>> &A_block) {
  profiler::func();

  using Dimensions = BlockDims<N, N>;
  std::array<f64, N * N> a_spread;

  for (u32 t = 0; t < steps; t++) {
    a_spread = A_block.Spread();

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        // Calculate indices
        // Row i-1
        u32 im1_jm1 = (i - 1) * N + (j - 1);
        u32 im1_j   = (i - 1) * N + j;
        u32 im1_jp1 = (i - 1) * N + (j + 1);
        // Row i
        u32 i_jm1   = i * N + (j - 1);
        u32 i_j     = i * N + j;
        u32 i_jp1   = i * N + (j + 1);
        // Row i+1
        u32 ip1_jm1 = (i + 1) * N + (j - 1);
        u32 ip1_j   = (i + 1) * N + j;
        u32 ip1_jp1 = (i + 1) * N + (j + 1);

        a_spread[i_j] = (a_spread[im1_jm1] + a_spread[im1_j] + a_spread[im1_jp1] +
                         a_spread[i_jm1]   + a_spread[i_j]   + a_spread[i_jp1] +
                         a_spread[ip1_jm1] + a_spread[ip1_j] + a_spread[ip1_jp1]) /
                        9.0;
      }
    }
    A_block = TestingBlock<Dimensions>(a_spread);
  }
}

template <size_t N>
static void Seidel2DSpreadBlockOnce(const int steps,
                                    TestingBlock<BlockDims<N, N>> &A_block) {
  profiler::func();

  using Dimensions = BlockDims<N, N>;

  std::array<f64, N * N> a_spread = A_block.Spread();

  for (u32 t = 0; t < steps; t++) {
    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        // Calculate indices
        // Row i-1
        u32 im1_jm1 = (i - 1) * N + (j - 1);
        u32 im1_j   = (i - 1) * N + j;
        u32 im1_jp1 = (i - 1) * N + (j + 1);
        // Row i
        u32 i_jm1   = i * N + (j - 1);
        u32 i_j     = i * N + j;
        u32 i_jp1   = i * N + (j + 1);
        // Row i+1
        u32 ip1_jm1 = (i + 1) * N + (j - 1);
        u32 ip1_j   = (i + 1) * N + j;
        u32 ip1_jp1 = (i + 1) * N + (j + 1);

        a_spread[i_j] = (a_spread[im1_jm1] + a_spread[im1_j] + a_spread[im1_jp1] +
                         a_spread[i_jm1]   + a_spread[i_j]   + a_spread[i_jp1] +
                         a_spread[ip1_jm1] + a_spread[ip1_j] + a_spread[ip1_jp1]) /
                        9.0;
      }
    }
  }

  A_block = TestingBlock<Dimensions>(a_spread);
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

  // Create and fill array
  auto a_base = std::array<std::array<f64, N>, N>{};

  for (u32 i = 0; i < N; i++) {
    a_base[i] = fill_random_arrays<f64, N>(-10, 10);
  }

  // Linearize for blocks
  std::array<f64, N * N> aLinear_base = {};
  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      aLinear_base[i * N + j] = a_base[i][j];
    }
  }

  // Get reference result
  auto a_ref = a_base;
  Seidel2DArray<N>(Steps, a_ref);

  profiler::begin();

  auto a_prof = a_base;
  Seidel2DArray<N>(Steps, a_prof); // Profiling reference too if needed, though usually just blocked ones

  Block blockA_naive(aLinear_base);
  Seidel2DNaiveBlock<N>(Steps, blockA_naive);

  Block blockA_spread_each(aLinear_base);
  Seidel2DSpreadBlockEach<N>(Steps, blockA_spread_each);

  Block blockA_spread_once(aLinear_base);
  Seidel2DSpreadBlockOnce<N>(Steps, blockA_spread_once);

  auto norm_ref = L2Norm(a_base); // Normalization against original base? Or Result?
  // Jacobi uses L2Norm(a_base), which is the norm of the *input*.
  // Let's stick to that.

  const auto collect_error_percent = [&](const f64 error_abs) {
    return (error_abs / norm_ref) * 100.0;
  };

  const auto error_naive = L2Norm<N>(a_ref, blockA_naive.Spread());
  const auto error_spread_each = L2Norm<N>(a_ref, blockA_spread_each.Spread());
  const auto error_spread_once = L2Norm<N>(a_ref, blockA_spread_once.Spread());

  return Iteration{
      ElementWise{collect_error_percent(error_naive),
                  collect_error_percent(error_spread_each),
                  collect_error_percent(error_spread_once)},
      ElementWise{error_naive, error_spread_each, error_spread_once},
  };
}

int main() {
  using Size = BlockDims<N, N>;
  using Block = TestingBlock<Size>;
  auto writer = CsvWriter();
  for (u32 Steps : StepsArray) {
    CsvInfo primitive = PrepareCsvPrimitive("seidel2d:primitive", N, Steps);
    CsvInfo block = PrepareCsvBlock<Block>("seidel2d:block", N, Steps);

    profiler::begin();

    for (int i = 0; i < Iterations; i++) {
      auto [percentage, absolute] = Test(Steps);

      writer.next_iteration();
      auto infos = profiler::dump_and_reset();

      for (auto &x : infos) {
        auto const &label = std::string(x.label);

        if (label == "Seidel2DArray") {
          writer.append_csv(primitive, x, 0, 0);
        } else if (label == "Seidel2DNaiveBlock") {
          writer.append_csv(block, x, percentage.naive, absolute.naive);
        } else if (label == "Seidel2DSpreadBlockEach") {
          writer.append_csv(block, x, percentage.spread_each,
                            absolute.spread_each);
        } else if (label == "Seidel2DSpreadBlockOnce") {
          writer.append_csv(block, x, percentage.spread_once,
                            absolute.spread_once);
        }
      }
    }
  }

  writer.dump("seidel2d.csv");

  return 0;
}
