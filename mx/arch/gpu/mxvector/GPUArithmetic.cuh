#pragma once
#include "../common.cuh"
#include <vector>
#include <cuda_runtime.h>

template <typename T>
T AddPointwiseGPUWrapper(const T& a, const T& b) {
    int blockSize = a.NumBlockElements() * sizeof(ElemType);
    size_t N = a.Size() * a.NumBlockElements();
    std::vector<ElemType> results (N);
    ElemType *d_l, *d_r, *d_result;
    cudaMalloc(&d_l, N * sizeof(ElemType));
    cudaMalloc(&d_r, N * sizeof(ElemType));
    cudaMalloc(&d_result, N * sizeof(ElemType));

    for (int i = 0; i < a.NumBlocks(); i++) {
        auto lhs = a.BlockAt(i);
        auto rhs = b.BlockAt(i);
        auto l = lhs.Spread();
        auto r = rhs.Spread();

        cudaMemcpy(d_l + i * a.NumBlockElements(), l.data(), blockSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r + i * b.NumBlockElements(), r.data(), blockSize, cudaMemcpyHostToDevice);
    }
    LaunchArithmeticKernel(d_l, d_r, d_result, N, ArithmeticOp::Add);
    for (int i = 0; i < a.NumBlocks(); i++) {
        cudaMemcpy(&results[i * a.NumBlockElements()], d_result + i * a.NumBlockElements(), blockSize, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_l);
    cudaFree(d_r);
    cudaFree(d_result);

    return T(results);
}