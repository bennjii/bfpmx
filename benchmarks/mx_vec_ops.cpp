#include <benchmark/benchmark.h>
#include "definition/vector/MxVectorOperations.hpp"
#include "definition/vector/MxVector.hpp"
#include "definition/prelude.h"
#include "arch/gpu/preludeGPU.cuh"
#include <random>

using MxVector = mx::vector::MxVector<BlockDims<32>, unsigned char, 
    fp8::E4M3Type, GPUArithmeticNaive, MaximumFractionalQuantization>;

static void BM_AddStdVector(benchmark::State& state) {
    const size_t n = state.range(0);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    std::vector<double> v1(n), v2(n), res(n);
    for (size_t i = 0; i < n; ++i) {
        v1[i] = dist(rng);
        v2[i] = dist(rng);
    }

    for (auto _ : state) {
        for (size_t i = 0; i < n; ++i) res[i] += v1[i] * v2[i];
        benchmark::DoNotOptimize(res);
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n * sizeof(double) * 2));
}

static void BM_AddMxVectorPointwiseGPU(benchmark::State& state) {
    const size_t n = state.range(0);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    std::vector<double> v1(n), v2(n);
    for (size_t i = 0; i < n; ++i) {
        v1[i] = dist(rng);
        v2[i] = dist(rng);
    }

    MxVector mx1(v1), mx2(v2);

    for (auto _ : state) {
        auto r = mx::vector::ops::AddPointwiseGPU(mx1, mx2);
        benchmark::DoNotOptimize(r);
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n * sizeof(double) * 2));
}

static void BM_AddMxVectorBlockwise(benchmark::State& state) {
    const size_t n = state.range(0);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    std::vector<double> v1(n), v2(n);
    for (size_t i = 0; i < n; ++i) {
        v1[i] = dist(rng);
        v2[i] = dist(rng);
    }

    MxVector mx1(v1), mx2(v2);

    for (auto _ : state) {
        auto r = mx::vector::ops::AddBlockwise(mx1, mx2);
        benchmark::DoNotOptimize(r);
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n * sizeof(double) * 2));
}

BENCHMARK(BM_AddStdVector)->Arg(10'000)->Arg(100'000)->Arg(1'000'000);
BENCHMARK(BM_AddMxVectorPointwiseGPU)->Arg(10'000)->Arg(100'000)->Arg(1'000'000);
BENCHMARK(BM_AddMxVectorBlockwise)->Arg(10'000)->Arg(100'000)->Arg(1'000'000);

BENCHMARK_MAIN();