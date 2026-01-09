#pragma once

#include "prelude.h"
#include <array>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>

namespace jacobi_benchmark {

// Convert MxVector to linear array
template <size_t N, typename MxVectorT>
void MxVectorToLinear(const MxVectorT &mxvec, std::array<f64, N * N> &linear) {
  const size_t limit = std::min(static_cast<size_t>(N * N), mxvec.Size());
  for (size_t idx = 0; idx < limit; idx++) {
    linear[idx] = mxvec.ItemAt(idx);
  }
}

// L2 norm computation
template <size_t N>
inline f64 L2Norm(const std::array<f64, N * N>& arr) {
  f64 sumSq = 0.0;
  for (size_t i = 0; i < N * N; i++) {
    sumSq += arr[i] * arr[i];
  }
  return std::sqrt(sumSq);
}

// L2 error between two linear arrays
template <size_t N>
inline f64 L2Error(const std::array<f64, N * N>& A, const std::array<f64, N * N>& B) {
  f64 sumSq = 0.0;
  for (size_t i = 0; i < N * N; i++) {
    f64 diff = A[i] - B[i];
    sumSq += diff * diff;
  }
  return std::sqrt(sumSq);
}

// Relative error as percentage: (L2 error / L2 norm of reference) * 100%
template <size_t N>
inline f64 RelativeErrorPercent(const std::array<f64, N * N>& reference, 
                                 const std::array<f64, N * N>& computed) {
  f64 l2Error = L2Error<N>(reference, computed);
  f64 l2NormRef = L2Norm<N>(reference);
  
  // Avoid division by zero
  if (l2NormRef == 0.0) {
    return (l2Error == 0.0) ? 0.0 : std::numeric_limits<f64>::infinity();
  }
  
  return (l2Error / l2NormRef) * 100.0;
}

// Mean absolute error
template <size_t N>
inline f64 MeanAbsError(const std::array<f64, N * N>& A, const std::array<f64, N * N>& B) {
  f64 sumAbs = 0.0;
  for (size_t i = 0; i < N * N; i++) {
    sumAbs += std::abs(A[i] - B[i]);
  }
  return sumAbs / (N * N);
}

// Max absolute error
template <size_t N>
inline f64 MaxAbsError(const std::array<f64, N * N>& A, const std::array<f64, N * N>& B) {
  f64 maxErr = 0.0;
  for (size_t i = 0; i < N * N; i++) {
    maxErr = std::max(maxErr, std::abs(A[i] - B[i]));
  }
  return maxErr;
}

// Aliases for backward compatibility with jacobi_gpu.cpp
template <size_t N>
inline f64 MaxAbsError2D(const std::array<f64, N * N>& A, const std::array<f64, N * N>& B) {
  return MaxAbsError<N>(A, B);
}

template <size_t N>
inline f64 MeanAbsError2D(const std::array<f64, N * N>& A, const std::array<f64, N * N>& B) {
  return MeanAbsError<N>(A, B);
}

// Convert 2D array to linear array (writes into pre-allocated output)
template <size_t N>
inline void Array2DToLinear(const std::array<std::array<f64, N>, N>& arr2D,
                            std::array<f64, N * N>& linear) {
  for (u32 i = 0; i < N; i++) {
    for (u32 j = 0; j < N; j++) {
      linear[i * N + j] = arr2D[i][j];
    }
  }
}

// Min absolute component value in array
template <size_t N>
inline f64 MinAbsComponent(const std::array<f64, N * N>& arr) {
  f64 minVal = std::numeric_limits<f64>::max();
  for (size_t i = 0; i < N * N; i++) {
    minVal = std::min(minVal, std::abs(arr[i]));
  }
  return minVal;
}

// Max absolute component value in array
template <size_t N>
inline f64 MaxAbsComponent(const std::array<f64, N * N>& arr) {
  f64 maxVal = 0.0;
  for (size_t i = 0; i < N * N; i++) {
    maxVal = std::max(maxVal, std::abs(arr[i]));
  }
  return maxVal;
}

// Max element-wise relative error: max_i |computed[i] - ref[i]| / |ref[i]|
// Only considers interior cells (skips boundary row 0, N-1 and col 0, N-1)
// Skips elements where ref[i] is very small to avoid division issues
template <size_t N>
inline f64 MaxElementRelativeError(const std::array<f64, N * N>& reference,
                                    const std::array<f64, N * N>& computed) {
  f64 maxRelErr = 0.0;
  constexpr f64 eps = 1;
  // Only check interior cells (skip boundaries)
  for (size_t row = 1; row < N - 1; row++) {
    for (size_t col = 1; col < N - 1; col++) {
      size_t i = row * N + col;
      f64 refAbs = std::abs(reference[i]);
      if (refAbs > eps && std::abs(computed[i]) > eps) {
        f64 relErr = std::abs(computed[i] - reference[i]) / refAbs;
        maxRelErr = std::max(maxRelErr, relErr);
      }
    }
  }
  return maxRelErr;
}

struct ImplementationResult {
  std::string name;
  f64 meanAbsError;
  f64 maxAbsError;
  f64 maxElemRelError;  // Max element-wise relative error: max_i |err[i]|/|ref[i]|
  f64 minAbsComponent;  // Min |x_i| in result vector (for error context)
  f64 maxAbsComponent;  // Max |x_i| in result vector (for error context)
  f64 timeTotalUs;      // Total time including allocation overhead
  f64 timeComputeUs;    // Compute-only time (excluding allocation)
};

struct BlockSizeResults {
  u32 blockSize;
  std::vector<ImplementationResult> results;
};

} // namespace jacobi_benchmark

