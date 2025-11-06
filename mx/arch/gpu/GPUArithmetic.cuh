#ifndef BFPMX_GPU_ARITHMETIC_H
#define BFPMX_GPU_ARITHMETIC_H

#include "definition/alias.h"
#include <array>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using ElemType = f64;
// Need a separate kernel because CUDA kernel launch can't be in cuh
void launchAddKernel(const ElemType* l, const ElemType* r, ElemType* out, size_t n);

template <typename T>
struct GPUArithmetic {

    static T Add(const T& lhs, const T& rhs) {
        constexpr size_t N = T::Length();

        auto l = lhs.Spread();
        auto r = rhs.Spread();
        std::array<ElemType, N> result;

        ElemType *d_l, *d_r, *d_result;
        cudaMalloc(&d_l, N * sizeof(ElemType));
        cudaMalloc(&d_r, N * sizeof(ElemType));
        cudaMalloc(&d_result, N * sizeof(ElemType));

        cudaMemcpy(d_l, l.data(), N * sizeof(ElemType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, r.data(), N * sizeof(ElemType), cudaMemcpyHostToDevice);

        launchAddKernel(d_l, d_r, d_result, N);

        cudaMemcpy(result.data(), d_result, N * sizeof(ElemType), cudaMemcpyDeviceToHost);

        cudaFree(d_l);
        cudaFree(d_r);
        cudaFree(d_result);

        return T(result);
    }
};

#endif
