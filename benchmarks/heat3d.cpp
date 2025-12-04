#define PROFILE 1

#include "prelude.h"
#include "profiler/profiler.h"

using TestingScalar = u32;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingBlock = Block<TestingScalar, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;

template <size_t N> using TestingMatrix3D = TestingBlock<BlockDims<N, N, N>>;

template <size_t N>
using NormalMatrix3D = std::array<std::array<std::array<f64, N>, N>, N>;

constexpr u32 N = 32; // grid size (32x32x32)

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

// MaxAbsError3D: Compute maximum absolute difference between two 3D arrays of
// the same size
template <size_t N>
f64 MaxAbsError3D(const NormalMatrix3D<N> A, const NormalMatrix3D<N> B) {
  f64 maxErr = 0.0;

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < N; k++) {
        maxErr = std::max(maxErr, std::abs(A[i][j][k] - B[i][j][k]));
      }
    }
  }

  return maxErr;
}

// MeanAbsError3D: Compute the avg element-wise absolute diff between 3D arrays
// of the same size
template <size_t N>
f64 MeanAbsError3D(const NormalMatrix3D<N> A, const NormalMatrix3D<N> B) {
  f64 sumAbs = 0.0;

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < N; k++) {
        sumAbs += std::abs(A[i][j][k] - B[i][j][k]);
      }
    }
  }

  return sumAbs / (N * N * N);
}

// ----------------------------------------
//    HEAT 3D STENCIL POLYBENCH VERSION
// ----------------------------------------
// ref:
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/stencils/heat-3d/heat-3d.c
template <size_t N>
static NormalMatrix3D<N> HeatReference3D(const int steps, NormalMatrix3D<N> A,
                                         NormalMatrix3D<N> B) {
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
  return A;
}

// ----------------------------------------
//    HEAT 3D STENCIL BLOCK VERSION
// ----------------------------------------
template <size_t N>
TestingMatrix3D<N> HeatBlock3D(const int steps, TestingMatrix3D<N> A,
                               TestingMatrix3D<N> B) {
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
              0.125 * (A(i + 1, j, k) - 2.0 * A(i, j, k) + A(i - 1, j, k)) +
              0.125 * (A(i, j + 1, k) - 2.0 * A(i, j, k) + A(i, j - 1, k)) +
              0.125 * (A(i, j, k + 1) - 2.0 * A(i, j, k) + A(i, j, k - 1)) +
              A(i, j, k);

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
              0.125 * (B(i + 1, j, k) - 2.0 * B(i, j, k) + B(i - 1, j, k)) +
              0.125 * (B(i, j + 1, k) - 2.0 * B(i, j, k) + B(i, j - 1, k)) +
              0.125 * (B(i, j, k + 1) - 2.0 * B(i, j, k) + B(i, j, k - 1)) +
              B(i, j, k);
          A.SetValue(idx, newVal);
          ++idx;
        }
      }
    }
  }
  return A;
}

f64 Test(const size_t steps) {
  using Size = BlockDims<N, N, N>;
  using Block = TestingBlock<Size>;

  NormalMatrix3D<N> a{}, b{};

  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      for (u32 k = 0; k < N; k++) {
        a[i][j][k] = (i + j + (N - k)) * 10.0 / N;
        b[i][j][k] = (i + j + (N - k)) * 10.0 / N;
      }
    }
  }

  // init linearized arrays for blocks
  std::array<f64, N * N * N> aLinear{};
  std::array<f64, N * N * N> bLinear{};

  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      for (u32 k = 0; k < N; k++) {
        aLinear[i * N * N + j * N + k] = a[i][j][k]; // aLinear[..] = a[..]
        bLinear[i * N * N + j * N + k] = b[i][j][k];
      }
    }
  }

  const auto blockA = Block(aLinear);
  const auto blockB = Block(bLinear);

  auto referenceResult = HeatReference3D<N>(steps, a, b);
  auto blockResult = HeatBlock3D<N>(steps, blockA, blockB);

  // Convert the block into an array so that we can calculate the error
  NormalMatrix3D<N> blockAsArray = BlockToArray3D<N>(blockResult);

  // calculate error(s)
  return MeanAbsError3D<N>(referenceResult, blockAsArray);
}

constexpr size_t Steps = 10;
constexpr size_t Iterations = 100;

int main() {
  f64 totalError = 0.0;

  profiler::begin();

  for (int i = 0; i < Iterations; i++) {
    totalError += Test(Steps);
  }

  f64 errorMean = totalError / Iterations;
  std::cout << "Mean absolute error: " << errorMean << std::endl;

  profiler::end_and_print();
}
