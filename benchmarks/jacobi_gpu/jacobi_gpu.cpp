//
// GPU Benchmark for Jacobi 2D using MxVector
// Tests 3 initialization scenarios to evaluate requantization behavior
//

#define PROFILE 1

#ifndef ENABLE_GPU_ARRAY
#define ENABLE_GPU_ARRAY 1
#endif
#ifndef ENABLE_GPU_MX_NAIVE
#define ENABLE_GPU_MX_NAIVE 1
#endif
#ifndef ENABLE_GPU_MX_BLOCKWISE
#define ENABLE_GPU_MX_BLOCKWISE 1
#endif
#ifndef ENABLE_GPU_MX_BLOCKWISE_REQUANT
#define ENABLE_GPU_MX_BLOCKWISE_REQUANT 1
#endif
#ifndef USE_CPU_REFERENCE
#define USE_CPU_REFERENCE 1
#endif

#include "prelude.h"
#include "arch/gpu/preludeGPU.cuh"
#include "JacobiKernel.cuh"
#include <nvtx3/nvToolsExt.h>
#include "definition/vector/MxVector.hpp"
#include "../jacobi_utils.h"
#include <iostream>
#include <memory>
#include <fstream>
#include <chrono>
#include <random>

using namespace jacobi_benchmark;

constexpr u32 N = 1024;
constexpr u32 Steps = 100;
constexpr u32 Iterations = 50;  // Number of iterations per configuration

using TestingScalar = unsigned char;
using TestingFloat = fp8::E4M3Type;
constexpr std::array<u32, 5> BlockSizes = {8, 16, 32, 64, 128};

using LinearField = std::array<f64, N * N>;

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

void InitializeField(LinearField& A, LinearField& B, InitScenario scenario, std::mt19937& rng) {
  switch (scenario) {
    case InitScenario::UniformShift:
      A.fill(10.0);
      B.fill(10.0);
      for (u32 i = 1; i < N - 1; i++) {
        for (u32 j = 1; j < N - 1; j++) {
          A[i * N + j] = 100.0;
        }
      }
      break;

    case InitScenario::RandomNarrow:
      {
        std::uniform_real_distribution<f64> dist(10.0, 20.0);
        for (u32 i = 0; i < N * N; i++) {
          A[i] = dist(rng);
        }
        B.fill(0.0);
      }
      break;

    case InitScenario::HighVariance:
      A.fill(0.0);
      B.fill(0.0);
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

// CPU Jacobi2D implementation
template <size_t N>
static void Jacobi2DLinear(const int steps, std::array<f64, N * N>& A, std::array<f64, N * N>& B) {
  for (int t = 0; t < steps; t++) {
    // A -> B
    for (u32 i = 1; i < N - 1; i++)
      for (u32 j = 1; j < N - 1; j++)
        B[i * N + j] = 0.2 * (A[i * N + j] + A[i * N + (j - 1)] + A[i * N + (j + 1)] + A[(i - 1) * N + j] + A[(i + 1) * N + j]);
    // B -> A
    for (u32 i = 1; i < N - 1; i++)
      for (u32 j = 1; j < N - 1; j++)
        A[i * N + j] = 0.2 * (B[i * N + j] + B[i * N + (j - 1)] + B[i * N + (j + 1)] + B[(i - 1) * N + j] + B[(i + 1) * N + j]);
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
             f64 mae, f64 maxAE, f64 maxElemRelErr, f64 minComp, f64 maxComp, f64 totalUs, f64 computeUs) {
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
                   const LinearField& aInit, const LinearField& bInit, const LinearField* ref) {
  using MxVectorT = mx::vector::MxVector<BlockDims<BlockSize>, TestingScalar, TestingFloat,
                                         GPUArithmeticNaive, MaximumFractionalQuantization>;
  
  std::vector<f64> aVec(aInit.begin(), aInit.end());
  std::vector<f64> bVec(bInit.begin(), bInit.end());

#if ENABLE_GPU_MX_NAIVE
  {
    MxVectorT gpuA(aVec), gpuB(bVec);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = Jacobi2DGPUMxVectorNaive(gpuA, gpuB, N, Steps);
    auto t1 = std::chrono::high_resolution_clock::now();
    f64 totalUs = static_cast<f64>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    
    MxVectorT gpuA2(aVec), gpuB2(bVec);
    Jacobi2DMxVectorNaiveContext<MxVectorT> ctx;
    Jacobi2DMxVectorNaiveSetup(ctx, gpuA2, gpuB2, N);
    f64 computeUs = timeUs([&]{ Jacobi2DMxVectorNaiveCompute(ctx, Steps); });
    Jacobi2DMxVectorNaiveTeardown(ctx);
    
    auto resultArr = std::make_unique<LinearField>();
    MxVectorToLinear<N>(result, *resultArr);
    
    f64 mae = ref ? MeanAbsError<N>(*ref, *resultArr) : 0;
    f64 maxAE = ref ? MaxAbsError<N>(*ref, *resultArr) : 0;
    f64 maxElemRelErr = ref ? MaxElementRelativeError<N>(*ref, *resultArr) : 0;
    csv.write(scenario, iter, BlockSize, "GPU MxVector SpreadOnce", mae, maxAE, maxElemRelErr,
              MinAbsComponent<N>(*resultArr), MaxAbsComponent<N>(*resultArr), totalUs, computeUs);
  }
#endif

#if ENABLE_GPU_MX_BLOCKWISE
  {
    MxVectorT gpuA(aVec), gpuB(bVec);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = Jacobi2DGPUMxVectorBlockWise(gpuA, gpuB, N, Steps);
    auto t1 = std::chrono::high_resolution_clock::now();
    f64 totalUs = static_cast<f64>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    
    MxVectorT gpuA2(aVec), gpuB2(bVec);
    Jacobi2DMxVectorBlockWiseContext<MxVectorT> ctx;
    Jacobi2DMxVectorBlockWiseSetup(ctx, gpuA2, gpuB2, N);
    f64 computeUs = timeUs([&]{ Jacobi2DMxVectorBlockWiseCompute(ctx, Steps); });
    Jacobi2DMxVectorBlockWiseTeardown(ctx);
    
    auto resultArr = std::make_unique<LinearField>();
    MxVectorToLinear<N>(result, *resultArr);
    
    f64 mae = ref ? MeanAbsError<N>(*ref, *resultArr) : 0;
    f64 maxAE = ref ? MaxAbsError<N>(*ref, *resultArr) : 0;
    f64 maxElemRelErr = ref ? MaxElementRelativeError<N>(*ref, *resultArr) : 0;
    csv.write(scenario, iter, BlockSize, "GPU MxVector BlockWise", mae, maxAE, maxElemRelErr,
              MinAbsComponent<N>(*resultArr), MaxAbsComponent<N>(*resultArr), totalUs, computeUs);
  }
#endif

#if ENABLE_GPU_MX_BLOCKWISE_REQUANT
  {
    MxVectorT gpuA(aVec), gpuB(bVec);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = Jacobi2DGPUMxVectorBlockWiseRequant(gpuA, gpuB, N, Steps);
    auto t1 = std::chrono::high_resolution_clock::now();
    f64 totalUs = static_cast<f64>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    
    MxVectorT gpuA2(aVec), gpuB2(bVec);
    Jacobi2DMxVectorBlockWiseRequantContext<MxVectorT> ctx;
    Jacobi2DMxVectorBlockWiseRequantSetup(ctx, gpuA2, gpuB2, N);
    f64 computeUs = timeUs([&]{ Jacobi2DMxVectorBlockWiseRequantCompute(ctx, Steps); });
    Jacobi2DMxVectorBlockWiseRequantTeardown(ctx);
    
    auto resultArr = std::make_unique<LinearField>();
    MxVectorToLinear<N>(result, *resultArr);
    
    f64 mae = ref ? MeanAbsError<N>(*ref, *resultArr) : 0;
    f64 maxAE = ref ? MaxAbsError<N>(*ref, *resultArr) : 0;
    f64 maxElemRelErr = ref ? MaxElementRelativeError<N>(*ref, *resultArr) : 0;
    csv.write(scenario, iter, BlockSize, "GPU MxVector BlockWiseRequant", mae, maxAE, maxElemRelErr,
              MinAbsComponent<N>(*resultArr), MaxAbsComponent<N>(*resultArr), totalUs, computeUs);
  }
#endif
}

void RunScenario(CSVWriter& csv, InitScenario scenario, std::mt19937& rng) {
  const char* scenarioName = ScenarioName(scenario);
  std::cout << "\n========================================\n";
  std::cout << "SCENARIO: " << scenarioName << "\n";
  std::cout << "========================================\n";

  // Initialize data for this scenario
  auto aInit = std::make_unique<LinearField>();
  auto bInit = std::make_unique<LinearField>();
  InitializeField(*aInit, *bInit, scenario, rng);

  // Compute CPU reference
  std::unique_ptr<LinearField> reference;
#if USE_CPU_REFERENCE
  auto aCpu = std::make_unique<LinearField>(*aInit);
  auto bCpu = std::make_unique<LinearField>(*bInit);
  Jacobi2DLinear<N>(Steps, *aCpu, *bCpu);
  reference = std::make_unique<LinearField>(*aCpu);
  std::cout << "Reference computed. Value range: [" 
            << MinAbsComponent<N>(*reference) << ", " << MaxAbsComponent<N>(*reference) << "]\n";
#endif

  // Run multiple iterations
  for (u32 iter = 0; iter < Iterations; iter++) {
    std::cout << "\n--- Iteration " << iter << " ---\n";
    
    // Test each block size
    for (u32 blockSize : BlockSizes) {
      std::cout << "  Block Size: " << blockSize << "\n";

#if USE_CPU_REFERENCE
      {
        auto aWork = std::make_unique<LinearField>(*aInit);
        auto bWork = std::make_unique<LinearField>(*bInit);
        f64 cpuUs = timeUs([&]{ Jacobi2DLinear<N>(Steps, *aWork, *bWork); });
        csv.write(scenarioName, iter, blockSize, "CPU Array", 0, 0, 0, 
                  MinAbsComponent<N>(*aWork), MaxAbsComponent<N>(*aWork), cpuUs, cpuUs);
      }
#endif

#if ENABLE_GPU_ARRAY
      {
        auto resultArr = std::make_unique<LinearField>();
        f64 totalUs = timeUs([&]{ Jacobi2DGPUArrayNaive(aInit->data(), bInit->data(), resultArr->data(), N, Steps); });
        
        Jacobi2DGPUArrayContext ctx;
        Jacobi2DGPUArraySetup(ctx, aInit->data(), bInit->data(), N);
        auto resultArr2 = std::make_unique<LinearField>();
        f64 computeUs = timeUs([&]{ Jacobi2DGPUArrayCompute(ctx, Steps); });
        Jacobi2DGPUArrayTeardown(ctx, resultArr2->data());
        
        f64 mae = reference ? MeanAbsError<N>(*reference, *resultArr) : 0;
        f64 maxAE = reference ? MaxAbsError<N>(*reference, *resultArr) : 0;
        f64 maxElemRelErr = reference ? MaxElementRelativeError<N>(*reference, *resultArr) : 0;
        csv.write(scenarioName, iter, blockSize, "GPU Array", mae, maxAE, maxElemRelErr,
                  MinAbsComponent<N>(*resultArr), MaxAbsComponent<N>(*resultArr), totalUs, computeUs);
      }
#endif

      switch (blockSize) {
        case 8:   TestBlockSize<8>(csv, scenarioName, iter, *aInit, *bInit, reference.get()); break;
        case 16:  TestBlockSize<16>(csv, scenarioName, iter, *aInit, *bInit, reference.get()); break;
        case 32:  TestBlockSize<32>(csv, scenarioName, iter, *aInit, *bInit, reference.get()); break;
        case 64:  TestBlockSize<64>(csv, scenarioName, iter, *aInit, *bInit, reference.get()); break;
        case 128: TestBlockSize<128>(csv, scenarioName, iter, *aInit, *bInit, reference.get()); break;
        case 256: TestBlockSize<256>(csv, scenarioName, iter, *aInit, *bInit, reference.get()); break;
      }
    }
  }
}

int main() {
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  CSVWriter csv("jacobi_gpu_results.csv");

  std::cout << "Jacobi2D Benchmark - 3 Initialization Scenarios\n";
  std::cout << "N=" << N << ", Steps=" << Steps << ", Iterations=" << Iterations << "\n";
  std::cout << "Block sizes: ";
  for (auto bs : BlockSizes) std::cout << bs << " ";
  std::cout << "\n";

  // Run all 3 scenarios
  RunScenario(csv, InitScenario::UniformShift, rng);
  RunScenario(csv, InitScenario::RandomNarrow, rng);
  RunScenario(csv, InitScenario::HighVariance, rng);

  std::cout << "\n========================================\n";
  std::cout << "Results written to jacobi_gpu_results.csv\n";
  std::cout << "========================================\n";
  return 0;
}
