#define PROFILE 1

#include "prelude.h"
#include "profiler/profiler.h"
#include "profiler/csv_info.h"

using TestingScalar = u32;
using TestingFloat = fp8::E4M3Type;
constexpr std::array<u32,4> StepsArray = {1,5,10,40};
constexpr u32 Iterations = 100;

template <typename Dimensions>
using TestingBlock = Block<TestingScalar, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;

template <size_t N> using TestingMatrix3D = TestingBlock<BlockDims<N, N, N>>;

template <size_t N>
using NormalMatrix3D = std::array<std::array<std::array<f64, N>, N>, N>;

constexpr u32 N = 20; // grid size (NxNxN)

// ----------------------------------------
//          UTILITY FUNCTIONS
// ----------------------------------------
// BlockToArray3D: Convert a 3Dim blockedFP format into a 3D array of f64
template <size_t N>
NormalMatrix3D<N> BlockToArray3D(const TestingMatrix3D<N> block) {
  NormalMatrix3D<N> out{};

  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      for (u32 k = 0; k < N; k++) {
        const u32 linear = TestingMatrix3D<N>::Shape::CoordsToLinear({i, j, k});
        out[i][j][k] = block.RealizeAtUnsafe(linear);
      }
    }
  }

  return out;
}

// L2 Norm (2 inputs) 
template <size_t N_>
f64 L2Norm3D(const NormalMatrix3D<N_> &A,const NormalMatrix3D<N_> &B){
  f64 error = 0.0;
  for (size_t i = 0; i < N_; i++){
    for (size_t j = 0; j < N_; j++){
      for (size_t k = 0; k < N_; k++){
        f64 diff = A[i][j][k] - B[i][j][k];
        error += diff * diff;
      }
    }
  }
  return std::sqrt(error);
}
// L2 Norm (1 input)
template <size_t N_>
static f64 L2Norm3D(const NormalMatrix3D<N_> &A){
  f64 norm_sq = 0.0;
  for (size_t i = 0; i < N_; i++){
    for (size_t j = 0; j < N_; j++){
      for (size_t k = 0; k < N_; k++){
        norm_sq += A[i][j][k] * A[i][j][k];
      }
    }
  }
  return std::sqrt(norm_sq);
}

// ----------------------------------------
//   (1) HEAT 3D STENCIL POLYBENCH VERSION
// ----------------------------------------
// ref:
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/stencils/heat-3d/heat-3d.c
template <size_t N>
static void Heat3DReference(const int steps, NormalMatrix3D<N> &A,
                                         NormalMatrix3D<N> &B) {
  profiler::func();
  for (int t = 1; t <= steps; t++) {
    for (int i = 1; i < N - 1; i++) {
      for (int j = 1; j < N - 1; j++) {
        for (int k = 1; k < N - 1; k++) {
          B[i][j][k] =
              0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k]) +
              0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k]) +
              0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1]) +
              A[i][j][k];
        }
      }
    }

    for (int i = 1; i < N - 1; i++) {
      for (int j = 1; j < N - 1; j++) {
        for (int k = 1; k < N - 1; k++) {
          A[i][j][k] =
              0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k]) +
              0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k]) +
              0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1]) +
              B[i][j][k];
        }
      }
    }
  }
}

// ----------------------------------------
//   (2) HEAT 3D STENCIL NAIVE BLOCK VERSION
// ----------------------------------------
template <size_t N>
static void Heat3DNaiveBlock(const int steps, TestingMatrix3D<N> &A,
                               TestingMatrix3D<N> &B) {
  profiler::func();

  for (u32 t = 0; t < steps; t++) {
    const u32 stride_j = N;
    const u32 stride_i = N * N;

    for (u32 i = 1; i < N - 1; i++) {
      const u32 offset_i = i * stride_i;
      for (u32 j = 1; j < N - 1; j++) {
        u32 idx = offset_i + (j * stride_j) + 1;
        for (u32 k = 1; k < N - 1; k++) {
          f64 newVal =
              0.125 * (A[i + 1, j, k] - 2.0 * A[i, j, k] + A[i - 1, j, k]) +
              0.125 * (A[i, j + 1, k] - 2.0 * A[i, j, k] + A[i, j - 1, k]) +
              0.125 * (A[i, j, k + 1] - 2.0 * A[i, j, k] + A[i, j, k - 1]) +
              A[i, j, k];

          B.SetValue(idx, newVal);
          ++idx;
        }
      }
    }

    for (u32 i = 1; i < N - 1; i++) {
      const u32 offset_i = i * stride_i;
      for (u32 j = 1; j < N - 1; j++) {
        u32 idx = offset_i + (j * stride_j) + 1;
        for (u32 k = 1; k < N - 1; k++) {
          f64 newVal =
              0.125 * (B[i + 1, j, k] - 2.0 * B[i, j, k] + B[i - 1, j, k]) +
              0.125 * (B[i, j + 1, k] - 2.0 * B[i, j, k] + B[i, j - 1, k]) +
              0.125 * (B[i, j, k + 1] - 2.0 * B[i, j, k] + B[i, j, k - 1]) +
              B[i, j, k];
          A.SetValue(idx, newVal);
          ++idx;
        }
      }
    }
  }
}

// -----------------------------------------------
//   (3) HEAT 3D STENCIL SPREADBLOCKEACH VERSION
// -----------------------------------------------
template <size_t N>
static void Heat3DSpreadBlockEach(const int steps,
                                                TestingMatrix3D<N> &A_block,
                                                TestingMatrix3D<N> &B_block) {
  profiler::func();
  std::array<f64, N * N * N> a_spread, b_spread;

  for (u32 t = 0; t < steps; t++) {
    a_spread = A_block.Spread();
    std::array<f64, N * N * N> b_new = a_spread;

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        for (u32 k = 1; k < N - 1; k++) {
          const u32 center = i * N * N + j * N + k;
          const u32 ip = (i + 1) * N * N + j * N + k;
          const u32 im = (i - 1) * N * N + j * N + k;
          const u32 jp = i * N * N + (j + 1) * N + k;
          const u32 jm = i * N * N + (j - 1) * N + k;
          const u32 kp = i * N * N + j * N + (k + 1);
          const u32 km = i * N * N + j * N + (k - 1);

          b_new[center] =
              0.125 * (a_spread[ip] - 2.0 * a_spread[center] + a_spread[im]) +
              0.125 * (a_spread[jp] - 2.0 * a_spread[center] + a_spread[jm]) +
              0.125 * (a_spread[kp] - 2.0 * a_spread[center] + a_spread[km]) +
              a_spread[center];
        }
      }
    }
    B_block = TestingMatrix3D<N>(b_new);

    b_spread = B_block.Spread();
    std::array<f64, N * N * N> a_new = b_spread;

    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        for (u32 k = 1; k < N - 1; k++) {
          const u32 center = i * N * N + j * N + k;
          const u32 ip = (i + 1) * N * N + j * N + k;
          const u32 im = (i - 1) * N * N + j * N + k;
          const u32 jp = i * N * N + (j + 1) * N + k;
          const u32 jm = i * N * N + (j - 1) * N + k;
          const u32 kp = i * N * N + j * N + (k + 1);
          const u32 km = i * N * N + j * N + (k - 1);

          a_new[center] =
              0.125 * (b_spread[ip] - 2.0 * b_spread[center] + b_spread[im]) +
              0.125 * (b_spread[jp] - 2.0 * b_spread[center] + b_spread[jm]) +
              0.125 * (b_spread[kp] - 2.0 * b_spread[center] + b_spread[km]) +
              b_spread[center];
        }
      }
    }
    A_block = TestingMatrix3D<N>(a_new);
  }
}


// -----------------------------------------------
//   (4) HEAT 3D STENCIL SPREADBLOCKONCE VERSION
// -----------------------------------------------
template <size_t N>
static void Heat3DSpreadBlockOnce(const int steps,
                                                TestingMatrix3D<N> &A_block,
                                                TestingMatrix3D<N> &B_block) {
  profiler::func();

  std::array<f64, N * N * N> a_spread = A_block.Spread();
  std::array<f64, N * N * N> b_spread = B_block.Spread();

  for (u32 t = 0; t < steps; t++) {
    // compute b_spread from a_spread
    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        for (u32 k = 1; k < N - 1; k++) {
          const u32 center = i * N * N + j * N + k;
          const u32 ip = (i + 1) * N * N + j * N + k;
          const u32 im = (i - 1) * N * N + j * N + k;
          const u32 jp = i * N * N + (j + 1) * N + k;
          const u32 jm = i * N * N + (j - 1) * N + k;
          const u32 kp = i * N * N + j * N + (k + 1);
          const u32 km = i * N * N + j * N + (k - 1);

          b_spread[center] =
              0.125 * (a_spread[ip] - 2.0 * a_spread[center] + a_spread[im]) +
              0.125 * (a_spread[jp] - 2.0 * a_spread[center] + a_spread[jm]) +
              0.125 * (a_spread[kp] - 2.0 * a_spread[center] + a_spread[km]) +
              a_spread[center];
        }
      }
    }

    // compute a_spread from b_spread
    for (u32 i = 1; i < N - 1; i++) {
      for (u32 j = 1; j < N - 1; j++) {
        for (u32 k = 1; k < N - 1; k++) {
          const u32 center = i * N * N + j * N + k;
          const u32 ip = (i + 1) * N * N + j * N + k;
          const u32 im = (i - 1) * N * N + j * N + k;
          const u32 jp = i * N * N + (j + 1) * N + k;
          const u32 jm = i * N * N + (j - 1) * N + k;
          const u32 kp = i * N * N + j * N + (k + 1);
          const u32 km = i * N * N + j * N + (k - 1);

          a_spread[center] =
              0.125 * (b_spread[ip] - 2.0 * b_spread[center] + b_spread[im]) +
              0.125 * (b_spread[jp] - 2.0 * b_spread[center] + b_spread[jm]) +
              0.125 * (b_spread[kp] - 2.0 * b_spread[center] + b_spread[km]) +
              b_spread[center];
        }
      }
    }
  }

  A_block = TestingMatrix3D<N>(a_spread);
  B_block = TestingMatrix3D<N>(b_spread);
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



Iteration Test(u32 steps) {
  using Size = BlockDims<N, N, N>;
  using Block = TestingBlock<Size>;

  NormalMatrix3D<N> a_base{}, b_base{};
  // TODO: Do we want to populate a_base/b_base each time randomly 
  //       or fix it such that it depends on N and not on ITERATION ?
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<> distrib(-10.0, 10.0);

  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      for (u32 k = 0; k < N; k++) {
        f64 random_val = distrib(gen); 
        a_base[i][j][k] = random_val; // fixed becomes: a[i][j][k] = (i + j + (N - k)) * 10.0 / N;
        b_base[i][j][k] = random_val; // similarly:     b[i][j][k] = (i + j + (N - k)) * 10.0 / N;
      }
    }
  }

  // init linearized arrays for blocks
  std::array<f64, N * N * N> aLinear_base{};
  std::array<f64, N * N * N> bLinear_base{};

  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      for (u32 k = 0; k < N; k++) {
        aLinear_base[i * N * N + j * N + k] = a_base[i][j][k]; // aLinear[..] = a[..]
        bLinear_base[i * N * N + j * N + k] = b_base[i][j][k];
      }
    }
  }

  auto a_ref = a_base;
  auto b_ref = b_base;

  profiler::begin();

  // Baseline (1)
  Heat3DReference<N>(steps, a_ref, b_ref);

  // NaiveBlock (2)
  Block blockA_naive(aLinear_base), blockB_naive(bLinear_base);
  Heat3DNaiveBlock<N>(steps, blockA_naive, blockB_naive);

  // SpreadBlockEach (3)
  Block blockA_spread_each(aLinear_base), blockB_spread_each(bLinear_base);
  Heat3DSpreadBlockEach<N>(steps, blockA_spread_each, blockB_spread_each);

  // SpreadBlockOnce (4)
  Block blockA_spread_once(aLinear_base), blockB_spread_once(bLinear_base);
  Heat3DSpreadBlockOnce<N>(steps, blockA_spread_once, blockB_spread_once);


  auto norm_ref = L2Norm3D(a_ref);


  const auto collect_error_percent = [&](const f64 error_abs) {
    return (error_abs / norm_ref) * 100.0;
  };


  const auto error_naive = L2Norm3D<N>(a_ref, BlockToArray3D<N>(blockA_naive));
  const auto error_spread_each = L2Norm3D<N>(a_ref, BlockToArray3D<N>(blockA_spread_each));
  const auto error_spread_once = L2Norm3D<N>(a_ref, BlockToArray3D<N>(blockA_spread_once));

  return Iteration{
      ElementWise{collect_error_percent(error_naive),
                  collect_error_percent(error_spread_each),
                  collect_error_percent(error_spread_once)},
      ElementWise{error_naive, error_spread_each, error_spread_once},
  };
}


int main() {
  using Size = BlockDims<N, N, N>;
  using Block = TestingBlock<Size>;
  auto writer = CsvWriter();
  for (u32 steps: StepsArray){
    CsvInfo primitive = PrepareCsvPrimitive("heat3d:primitive", N, steps);
    CsvInfo block = PrepareCsvBlock<Block>("heat3d:block", N, steps);


    profiler::begin();

    for (int i = 0; i < Iterations; i++) {
      auto [percentage, absolute] = Test(steps);

      writer.next_iteration();
      auto infos = profiler::dump_and_reset();

      for (auto &x : infos) {
        auto const &label = std::string(x.label);

        if (label == "Heat3DReference") {
          writer.append_csv(primitive, x, 0, 0);
        } else if (label == "Heat3DNaiveBlock") {
          writer.append_csv(block, x, percentage.naive, absolute.naive);
        } else if (label == "Heat3DSpreadBlockEach") {
          writer.append_csv(block, x, percentage.spread_each,
                            absolute.spread_each);
        } else if (label == "Heat3DSpreadBlockOnce") {
          writer.append_csv(block, x, percentage.spread_once,
                            absolute.spread_once);
        }
      }
    }
  }

  writer.dump("../benchmarks/heat3d.csv");
  return 0;
}
