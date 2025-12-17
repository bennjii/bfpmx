//
// GPU Benchmark for Jacobi 2D using MxVector
// Similar to PolyBench test suite
//

#define PROFILE 1

#include "prelude.h"
#include "profiler/profiler.h"
#ifdef HAS_CUDA
#include "arch/gpu/preludeGPU.cuh"
#endif
#include "definition/vector/MxVector.hpp"
#include <cmath>
#include <limits>
#include <iostream>

constexpr u32 N = 300;
constexpr u32 Steps = 250;
constexpr u32 Iterations = 100;

using TestingScalar = unsigned char;
using TestingFloat = fp8::E4M3Type;

// GPU MxVector type for Jacobi benchmark
using GPUMxVector = mx::vector::MxVector<BlockDims<128>, TestingScalar, TestingFloat,
                                         GPUArithmeticNaive, MaximumFractionalQuantization>;

// L2 norm computation for arrays
template <size_t N>
f64 L2Norm(const std::array<f64, N * N>& arr) {
  f64 sumSq = 0.0;
  for (size_t i = 0; i < N * N; i++) {
    sumSq += arr[i] * arr[i];
  }
  return std::sqrt(sumSq);
}

// Error computation functions for 2D arrays
template <size_t N>
f64 MaxAbsError2D(const std::array<f64, N * N>& A, const std::array<f64, N * N>& B) {
  f64 maxErr = 0.0;
  for (size_t i = 0; i < N * N; i++) {
    maxErr = std::max(maxErr, std::abs(A[i] - B[i]));
  }
  return maxErr;
}

template <size_t N>
f64 MeanAbsError2D(const std::array<f64, N * N>& A, const std::array<f64, N * N>& B) {
  f64 sumAbs = 0.0;
  for (size_t i = 0; i < N * N; i++) {
    sumAbs += std::abs(A[i] - B[i]);
  }
  return sumAbs / (N * N);
}

// L2 norm of error vector
template <size_t N>
f64 L2Error(const std::array<f64, N * N>& A, const std::array<f64, N * N>& B) {
  f64 sumSq = 0.0;
  for (size_t i = 0; i < N * N; i++) {
    f64 diff = A[i] - B[i];
    sumSq += diff * diff;
  }
  return std::sqrt(sumSq);
}

// Relative error as percentage: (L2 error / L2 norm of reference) * 100%
template <size_t N>
f64 RelativeErrorPercent(const std::array<f64, N * N>& reference, 
                         const std::array<f64, N * N>& computed) {
  f64 l2Error = L2Error<N>(reference, computed);
  f64 l2NormRef = L2Norm<N>(reference);
  
  // Avoid division by zero
  if (l2NormRef == 0.0) {
    return (l2Error == 0.0) ? 0.0 : std::numeric_limits<f64>::infinity();
  }
  
  return (l2Error / l2NormRef) * 100.0;
}

// Convert 2D array to linear array
template <size_t N>
std::array<f64, N * N> Array2DToLinear(const std::array<std::array<f64, N>, N>& arr2D) {
  std::array<f64, N * N> linear = {};
  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      linear[i * N + j] = arr2D[i][j];
    }
  }
  return linear;
}

// Convert MxVector to linear array
template <size_t N>
std::array<f64, N * N> MxVectorToLinear(const GPUMxVector& mxvec) {
  std::array<f64, N * N> linear = {};
  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      const u32 idx = i * N + j;
      if (idx < mxvec.Size()) {
        linear[idx] = mxvec.ItemAt(idx);
      }
    }
  }
  return linear;
}

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

#ifdef HAS_CUDA
// GPU implementation using normal CUDA arrays (f64) - wrapper
template <size_t N>
static void Jacobi2DGPUArray(const int steps,
                             const std::array<f64, N * N>& A_host,
                             const std::array<f64, N * N>& B_host,
                             std::array<f64, N * N>& result) {
  profiler::func();
  Jacobi2DGPUArrayNaive(A_host.data(), B_host.data(), result.data(), N, steps);
}

// GPU implementation using MxVector
template <size_t N>
static GPUMxVector Jacobi2DGPUMxVector(const int steps,
                                       const GPUMxVector& A,
                                       const GPUMxVector& B) {
  profiler::func();
  return Jacobi2DGPUMxVectorNaive(A, B, N, steps);
}
#endif

struct ErrorResults {
  f64 gpuArrayMaxError;
  f64 gpuArrayMeanError;
  f64 gpuArrayRelativeError;
  f64 gpuMxMaxError;
  f64 gpuMxMeanError;
  f64 gpuMxRelativeError;
};

ErrorResults Test() {
  auto a = std::array<std::array<f64, N>, N>{
      std::array<f64, N>{1.2f},
      std::array<f64, N>{3.4f},
  };

  auto b = std::array<std::array<f64, N>, N>{
      std::array<f64, N>{1.2f},
      std::array<f64, N>{3.4f},
  };

  // CPU reference implementation
  auto a_cpu = a;
  auto b_cpu = b;
  Jacobi2DArray<N>(Steps, a_cpu, b_cpu);
  std::array<f64, N * N> cpuResult = Array2DToLinear<N>(a_cpu);

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
  auto black_box_array = [&](const std::array<f64, N * N>& arr) {
    f64 sum = 0;
    for (u32 i = 0; i < N * N; i++) {
      sum += arr[i];
    }
    volatile f64 x = sum;
  };

  // used to prevent compiler optimization on the calls
  auto black_box_mxvector = [&](const auto &a, const auto &b) {
    f64 sum = 0;
    for (u32 i = 0; i < N; i++) {
      for (u32 j = 0; j < N; j++) {
        const u32 idx = i * N + j;
        if (idx < a.Size() && idx < b.Size()) {
          sum += a.ItemAt(idx) + b.ItemAt(idx);
        }
      }
    }
    volatile f64 x = sum;
  };

#ifdef HAS_CUDA
  // GPU implementation with normal CUDA arrays
  std::array<f64, N * N> gpuArrayResult = {};
  Jacobi2DGPUArray<N>(Steps, aLinear, bLinear, gpuArrayResult);
  black_box_array(gpuArrayResult);

  // Compute errors for GPU array implementation
  f64 gpuArrayMaxError = MaxAbsError2D<N>(cpuResult, gpuArrayResult);
  f64 gpuArrayMeanError = MeanAbsError2D<N>(cpuResult, gpuArrayResult);
  f64 gpuArrayRelativeError = RelativeErrorPercent<N>(cpuResult, gpuArrayResult);

  // GPU implementation with MxVector
  // Convert std::array to std::vector for MxVector constructor
  std::vector<f64> aVec(aLinear.begin(), aLinear.end());
  std::vector<f64> bVec(bLinear.begin(), bLinear.end());

  GPUMxVector gpuA(aVec);
  GPUMxVector gpuB(bVec);

  auto gpuMxResult = Jacobi2DGPUMxVector<N>(Steps, gpuA, gpuB);
  black_box_mxvector(gpuMxResult, gpuB);

  // Convert MxVector result to linear array for comparison
  std::array<f64, N * N> gpuMxResultArray = MxVectorToLinear<N>(gpuMxResult);

  // Compute errors for GPU MxVector implementation
  f64 gpuMxMaxError = MaxAbsError2D<N>(cpuResult, gpuMxResultArray);
  f64 gpuMxMeanError = MeanAbsError2D<N>(cpuResult, gpuMxResultArray);
  f64 gpuMxRelativeError = RelativeErrorPercent<N>(cpuResult, gpuMxResultArray);

  // Return struct with errors for both implementations
  return ErrorResults{gpuArrayMaxError, gpuArrayMeanError, gpuArrayRelativeError,
                      gpuMxMaxError, gpuMxMeanError, gpuMxRelativeError};
#else
  return ErrorResults{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
#endif
}

int main() {
  profiler::begin();

  f64 maxGpuArrayMaxError = 0.0;
  f64 totalGpuArrayMeanError = 0.0;
  f64 totalGpuArrayRelativeError = 0.0;
  f64 maxGpuMxMaxError = 0.0;
  f64 totalGpuMxMeanError = 0.0;
  f64 totalGpuMxRelativeError = 0.0;

  for (int i = 0; i < Iterations; i++) {
    ErrorResults errors = Test();
    maxGpuArrayMaxError = std::max(maxGpuArrayMaxError, errors.gpuArrayMaxError);
    totalGpuArrayMeanError += errors.gpuArrayMeanError;
    totalGpuArrayRelativeError += errors.gpuArrayRelativeError;
    maxGpuMxMaxError = std::max(maxGpuMxMaxError, errors.gpuMxMaxError);
    totalGpuMxMeanError += errors.gpuMxMeanError;
    totalGpuMxRelativeError += errors.gpuMxRelativeError;
  }

  f64 avgGpuArrayMeanError = totalGpuArrayMeanError / Iterations;
  f64 avgGpuArrayRelativeError = totalGpuArrayRelativeError / Iterations;
  f64 avgGpuMxMeanError = totalGpuMxMeanError / Iterations;
  f64 avgGpuMxRelativeError = totalGpuMxRelativeError / Iterations;

  std::cout << "\n=== Error Analysis (vs CPU Reference) ===" << std::endl;
  std::cout << "GPU Array Implementation:" << std::endl;
  std::cout << "  Mean Absolute Error: " << avgGpuArrayMeanError << std::endl;
  std::cout << "  Max Absolute Error:  " << maxGpuArrayMaxError << std::endl;
  std::cout << "  Relative Error (L2): " << avgGpuArrayRelativeError << "%" << std::endl;
  std::cout << "GPU MxVector Implementation:" << std::endl;
  std::cout << "  Mean Absolute Error: " << avgGpuMxMeanError << std::endl;
  std::cout << "  Max Absolute Error:  " << maxGpuMxMaxError << std::endl;
  std::cout << "  Relative Error (L2): " << avgGpuMxRelativeError << "%" << std::endl;
  std::cout << "==========================================\n" << std::endl;

  profiler::end_and_print();
  return 0;
}

