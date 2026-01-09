//
// Benchmark for Jacobi 2D using C++ Standard Parallelism (stdpar) with MxVector
// Tests 3 initialization scenarios to evaluate requantization behavior
//

#include "prelude.h"
#include "helper/test.h"
#include "definition/vector/MxVector.hpp"
#include "jacobi_stdpar/JacobiStdPar.hpp"
#include "jacobi_utils.h"
#include <vector>
#include <array>
#include <algorithm>
#include <memory>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <limits>
#include <iostream>
#include <iomanip>
#include <random>

using namespace jacobi_benchmark;

constexpr u32 N = 1024;
constexpr u32 Steps = 100;
constexpr u32 Iterations = 50;  // Number of iterations per configuration

// Block sizes to test
constexpr std::array<u32, 5> BlockSizes = {8, 16, 32, 64, 128};

using TestingScalar = unsigned char;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingMxVector = mx::vector::MxVector<Dimensions, TestingScalar, TestingFloat,
                                              CPUArithmetic, MaximumFractionalQuantization>;

using LinearField = std::vector<f64>;

// Initialization scenarios
enum class InitScenario {
  UniformShift,   // Boundary=10, Interior=100 (MX-friendly, tests scalar adaptation)
  RandomNarrow,   // Random [10,20] everywhere (baseline, scalar stays similar)
  HighVariance    // Boundary=0, Interior=random[100,1000] (stress test, MX weakness)
};

const char* ScenarioName(InitScenario s) {
  switch (s) {
    case InitScenario::UniformShift: return "UniformShift";
    case InitScenario::RandomNarrow: return "RandomNarrow";
    case InitScenario::HighVariance: return "HighVariance";
  }
  return "Unknown";
}

// Initialize field based on scenario
void InitializeField(LinearField& A, LinearField& B, InitScenario scenario, std::mt19937& rng) {
  A.resize(N * N);
  B.resize(N * N);
  
  switch (scenario) {
    case InitScenario::UniformShift:
      // Boundary = 10, Interior = 100 (uniform)
      // Values smoothly decay from 100 toward 10, testing scalar adaptation
      std::fill(A.begin(), A.end(), 10.0);
      std::fill(B.begin(), B.end(), 10.0);
      for (u32 i = 1; i < N - 1; i++) {
        for (u32 j = 1; j < N - 1; j++) {
          A[i * N + j] = 100.0;
        }
      }
      break;

    case InitScenario::RandomNarrow:
      // Random values in [10, 20] everywhere
      // Scalar stays ~4 throughout, requant shows no benefit
      {
        std::uniform_real_distribution<f64> dist(10.0, 20.0);
        for (u32 i = 0; i < N * N; i++) {
          A[i] = dist(rng);
        }
        std::fill(B.begin(), B.end(), 0.0);
      }
      break;

    case InitScenario::HighVariance:
      // Boundary = 0, Interior = random [100, 1000]
      // High variance within blocks near boundary - stress test for MX
      std::fill(A.begin(), A.end(), 0.0);
      std::fill(B.begin(), B.end(), 0.0);
      {
        std::uniform_real_distribution<f64> dist(100.0, 1000.0);
        for (u32 i = 1; i < N - 1; i++) {
          for (u32 j = 1; j < N - 1; j++) {
            A[i * N + j] = dist(rng);
          }
        }
      }
      break;
  }
}

// Baseline: Sequential, non-parallel CPU implementation
static void Jacobi2DSequentialArray(const std::vector<f64>& aInit,
                                     const std::vector<f64>& bInit,
                                     std::vector<f64>& aResult,
                                     std::vector<f64>& bResult,
                                     u32 steps) {
  aResult = aInit;
  bResult = bInit;
  
  // Sequential Jacobi iterations (no parallelization)
  for (u32 t = 0; t < steps; ++t) {
    // First half-step: compute B from A (interior points only)
    for (u32 i = 1; i < N - 1; ++i) {
      for (u32 j = 1; j < N - 1; ++j) {
        u32 center = i * N + j;
        u32 left = i * N + (j - 1);
        u32 right = i * N + (j + 1);
        u32 up = (i - 1) * N + j;
        u32 down = (i + 1) * N + j;
        bResult[center] = 0.2 * (aResult[center] + aResult[left] + 
                                 aResult[right] + aResult[down] + aResult[up]);
      }
    }
    
    // Second half-step: compute A from B (interior points only)
    for (u32 i = 1; i < N - 1; ++i) {
      for (u32 j = 1; j < N - 1; ++j) {
        u32 center = i * N + j;
        u32 left = i * N + (j - 1);
        u32 right = i * N + (j + 1);
        u32 up = (i - 1) * N + j;
        u32 down = (i + 1) * N + j;
        aResult[center] = 0.2 * (bResult[center] + bResult[left] + 
                                 bResult[right] + bResult[down] + bResult[up]);
      }
    }
  }
}

// Reference implementation: StdPar on plain vectors (same implementation as MxVector)
static void Jacobi2DStdParArray(const std::vector<f64>& aInit,
                                const std::vector<f64>& bInit,
                                std::vector<f64>& aResult,
                                std::vector<f64>& bResult,
                                u32 steps) {
  aResult = aInit;
  bResult = bInit;
  
  // Run Jacobi iterations using stdpar (same as MxVector version)
  for (u32 t = 0; t < steps; ++t) {
    // First half-step: compute B from A
    mx::arch::gpu::stdpar::JacobiStencilStepStdPar(aResult, bResult, N);
    
    // Second half-step: compute A from B (swap arguments)
    mx::arch::gpu::stdpar::JacobiStencilStepStdPar(bResult, aResult, N);
  }
}

// CSV output helper
class CSVWriter {
  std::ofstream file;
public:
  CSVWriter(const std::string& filename) : file(filename) {
    file << "Scenario,Iteration,BlockSize,Implementation,MAE,MaxAE,MaxElemRelErr,MinAbsComp,MaxAbsComp,Time_total_us,Time_compute_us\n";
  }
  void write(const char* scenario, u32 iter, u32 blockSize, const std::string& name, 
             f64 mae, f64 maxAE, f64 maxElemRelErr, f64 minComp, f64 maxComp, 
             f64 totalUs, f64 computeUs) {
    file << scenario << "," << iter << "," << blockSize << "," << name << "," 
         << mae << "," << maxAE << "," << maxElemRelErr << ","
         << minComp << "," << maxComp << "," << totalUs << "," << computeUs << "\n";
  }
};

// Timing helper
template<typename F>
f64 timeUs(F&& func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  return static_cast<f64>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

// Test a specific block size for one iteration
template <u32 BlockSize>
void TestBlockSize(CSVWriter& csv, const char* scenario, u32 iter, 
                   const std::vector<f64>& aVec,
                   const std::vector<f64>& bVec,
                   const LinearField* reference) {
  using MxVectorT = TestingMxVector<BlockDims<BlockSize>>;

  // Convert reference vector to array for error functions
  auto refArray = std::make_unique<std::array<f64, N * N>>();
  if (reference) {
    for (u32 i = 0; i < N * N && i < reference->size(); ++i) {
      (*refArray)[i] = (*reference)[i];
    }
  }

  // Test MxVector stdpar implementation - SpreadOnce
  {
    MxVectorT mxA(aVec);
    MxVectorT mxB(bVec);

    auto t0 = std::chrono::high_resolution_clock::now();
    mx::arch::gpu::stdpar::Jacobi2DStdParSpreadOnce(mxA, mxB, N, Steps);
    auto t1 = std::chrono::high_resolution_clock::now();
    f64 totalUs = static_cast<f64>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    f64 computeUs = totalUs;  // For stdpar, total and compute are the same

    // Convert MxVector result to linear array for comparison
    auto mxResultArray = std::make_unique<std::array<f64, N * N>>();
    MxVectorToLinear<N>(mxA, *mxResultArray);

    // Compute errors
    f64 mae = reference ? MeanAbsError<N>(*refArray, *mxResultArray) : 0;
    f64 maxAE = reference ? MaxAbsError<N>(*refArray, *mxResultArray) : 0;
    f64 maxElemRelErr = reference ? MaxElementRelativeError<N>(*refArray, *mxResultArray) : 0;
    
    csv.write(scenario, iter, BlockSize, "GPU MxVector StdPar SpreadOnce", mae, maxAE, maxElemRelErr,
              MinAbsComponent<N>(*mxResultArray), MaxAbsComponent<N>(*mxResultArray), totalUs, computeUs);
  }

  // Test MxVector Fused stdpar implementation - OnTheFly
  {
    MxVectorT mxA(aVec);
    MxVectorT mxB(bVec);

    auto t0 = std::chrono::high_resolution_clock::now();
    mx::arch::gpu::stdpar::Jacobi2DStdParOnTheFly(mxA, mxB, N, Steps);
    auto t1 = std::chrono::high_resolution_clock::now();
    f64 totalUs = static_cast<f64>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    f64 computeUs = totalUs;  // For stdpar, total and compute are the same

    // Convert MxVector result to linear array for comparison
    auto mxResultArray = std::make_unique<std::array<f64, N * N>>();
    MxVectorToLinear<N>(mxA, *mxResultArray);

    // Compute errors
    f64 mae = reference ? MeanAbsError<N>(*refArray, *mxResultArray) : 0;
    f64 maxAE = reference ? MaxAbsError<N>(*refArray, *mxResultArray) : 0;
    f64 maxElemRelErr = reference ? MaxElementRelativeError<N>(*refArray, *mxResultArray) : 0;
    
    csv.write(scenario, iter, BlockSize, "GPU MxVector StdPar OnTheFly", mae, maxAE, maxElemRelErr,
              MinAbsComponent<N>(*mxResultArray), MaxAbsComponent<N>(*mxResultArray), totalUs, computeUs);
  }
}

void RunScenario(CSVWriter& csv, InitScenario scenario, std::mt19937& rng) {
  const char* scenarioName = ScenarioName(scenario);
  std::cout << "\n========================================\n";
  std::cout << "SCENARIO: " << scenarioName << "\n";
  std::cout << "========================================\n";

  // Initialize data for this scenario (use unique_ptr for heap allocation)
  auto aInit = std::make_unique<LinearField>();
  auto bInit = std::make_unique<LinearField>();
  InitializeField(*aInit, *bInit, scenario, rng);

  // Compute CPU reference
  auto reference = std::make_unique<LinearField>();
  {
    auto aCpu = std::make_unique<LinearField>(*aInit);
    auto bCpu = std::make_unique<LinearField>(*bInit);
    auto aRefVec = std::make_unique<std::vector<f64>>(N * N);
    auto bRefVec = std::make_unique<std::vector<f64>>(N * N);
    Jacobi2DSequentialArray(*aCpu, *bCpu, *aRefVec, *bRefVec, Steps);
    *reference = *aRefVec;
    
    // Convert to array for MinAbsComponent/MaxAbsComponent
    auto refArray = std::make_unique<std::array<f64, N * N>>();
    for (u32 i = 0; i < N * N && i < reference->size(); ++i) {
      (*refArray)[i] = (*reference)[i];
    }
    std::cout << "Reference computed. Value range: [" 
              << MinAbsComponent<N>(*refArray) << ", " << MaxAbsComponent<N>(*refArray) << "]\n";
  }

  // Run multiple iterations
  for (u32 iter = 0; iter < Iterations; iter++) {
    std::cout << "\n--- Iteration " << iter << " ---\n";
    
    // Test sequential baseline (error reference)
    {
      auto aVec = std::make_unique<std::vector<f64>>(*aInit);
      auto bVec = std::make_unique<std::vector<f64>>(*bInit);
      auto aResultVec = std::make_unique<std::vector<f64>>(N * N);
      auto bResultVec = std::make_unique<std::vector<f64>>(N * N);
      
      f64 cpuUs = timeUs([&]{ Jacobi2DSequentialArray(*aVec, *bVec, *aResultVec, *bResultVec, Steps); });
      
      auto resultArray = std::make_unique<std::array<f64, N * N>>();
      for (u32 i = 0; i < N * N && i < aResultVec->size(); ++i) {
        (*resultArray)[i] = (*aResultVec)[i];
      }
      
      csv.write(scenarioName, iter, 0, "Sequential Array", 0, 0, 0,
                MinAbsComponent<N>(*resultArray), MaxAbsComponent<N>(*resultArray), cpuUs, cpuUs);
    }

    // Test stdpar array (runtime baseline)
    {
      auto aVec = std::make_unique<std::vector<f64>>(*aInit);
      auto bVec = std::make_unique<std::vector<f64>>(*bInit);
      auto aStdparVec = std::make_unique<std::vector<f64>>(N * N);
      auto bStdparVec = std::make_unique<std::vector<f64>>(N * N);
      
      f64 stdparUs = timeUs([&]{ Jacobi2DStdParArray(*aVec, *bVec, *aStdparVec, *bStdparVec, Steps); });
      
      auto stdparArray = std::make_unique<std::array<f64, N * N>>();
      for (u32 i = 0; i < N * N && i < aStdparVec->size(); ++i) {
        (*stdparArray)[i] = (*aStdparVec)[i];
      }
      
      // Convert reference to array for error functions
      auto refArray = std::make_unique<std::array<f64, N * N>>();
      for (u32 i = 0; i < N * N && i < reference->size(); ++i) {
        (*refArray)[i] = (*reference)[i];
      }
      
      f64 mae = MeanAbsError<N>(*refArray, *stdparArray);
      f64 maxAE = MaxAbsError<N>(*refArray, *stdparArray);
      f64 maxElemRelErr = MaxElementRelativeError<N>(*refArray, *stdparArray);
      
      csv.write(scenarioName, iter, 0, "StdPar Array", mae, maxAE, maxElemRelErr,
                MinAbsComponent<N>(*stdparArray), MaxAbsComponent<N>(*stdparArray), stdparUs, stdparUs);
    }

    // Test each block size
    for (u32 blockSize : BlockSizes) {
      std::cout << "  Block Size: " << blockSize << "\n";
      
      auto aVec = std::make_unique<std::vector<f64>>(*aInit);
      auto bVec = std::make_unique<std::vector<f64>>(*bInit);

      switch (blockSize) {
        case 8:   TestBlockSize<8>(csv, scenarioName, iter, *aVec, *bVec, reference.get()); break;
        case 16:  TestBlockSize<16>(csv, scenarioName, iter, *aVec, *bVec, reference.get()); break;
        case 32:  TestBlockSize<32>(csv, scenarioName, iter, *aVec, *bVec, reference.get()); break;
        case 64:  TestBlockSize<64>(csv, scenarioName, iter, *aVec, *bVec, reference.get()); break;
        case 128: TestBlockSize<128>(csv, scenarioName, iter, *aVec, *bVec, reference.get()); break;
        default:
          std::cerr << "Unsupported block size: " << blockSize << std::endl;
          break;
      }
    }
  }
}

int main() {
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  CSVWriter csv("jacobi_stdpar_results.csv");

  std::cout << "Jacobi2D StdPar Benchmark - 3 Initialization Scenarios\n";
  std::cout << "N=" << N << ", Steps=" << Steps << ", Iterations=" << Iterations << "\n";
  std::cout << "Block sizes: ";
  for (auto bs : BlockSizes) std::cout << bs << " ";
  std::cout << "\n";

  // Run all 3 scenarios
  RunScenario(csv, InitScenario::UniformShift, rng);
  RunScenario(csv, InitScenario::RandomNarrow, rng);
  RunScenario(csv, InitScenario::HighVariance, rng);

  std::cout << "\n========================================\n";
  std::cout << "Results written to jacobi_stdpar_results.csv\n";
  std::cout << "========================================\n";
  return 0;
}
