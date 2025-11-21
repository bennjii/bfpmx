#include <benchmark/benchmark.h>
#include "definition/vector/MxVectorOperations.hpp"
#include "definition/vector/MxVector.hpp"
#include <random>

static void BM_DotStdVector(benchmark::State& state) {
    const size_t n = state.range(0);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    std::vector<double> v1(n), v2(n);
    for (size_t i = 0; i < n; ++i) {
        v1[i] = dist(rng);
        v2[i] = dist(rng);
    }

    for (auto _ : state) {
        double sum = 0;
        for (size_t i = 0; i < n; ++i) sum += v1[i] * v2[i];
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n * sizeof(double) * 2));
}

static void BM_DotMxVector(benchmark::State& state) {
    const size_t n = state.range(0);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    std::vector<double> v1(n), v2(n);
    for (size_t i = 0; i < n; ++i) {
        v1[i] = dist(rng);
        v2[i] = dist(rng);
    }

    mx::vector::MxVector<BlockDims<32>> mx1(v1), mx2(v2);

    for (auto _ : state) {
        double r = mx::vector::ops::Dot(mx1, mx2);
        benchmark::DoNotOptimize(r);
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n * sizeof(double) * 2));
}

BENCHMARK(BM_DotStdVector)->Arg(10'000)->Arg(100'000)->Arg(1'000'000);
BENCHMARK(BM_DotMxVector)->Arg(10'000)->Arg(100'000)->Arg(1'000'000);

BENCHMARK_MAIN();