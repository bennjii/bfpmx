// This header file must NOT be included in any non-CUDA file
#ifndef BFPMX_GPU_ARITHMETIC_H
#define BFPMX_GPU_ARITHMETIC_H

#include "definition/alias.h"
#include <array>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using ElemType = f64;
enum class ArithmeticOp : uint8_t {
    Add,
    Sub,
    Mul,
    Div
};

// Arithmetic kernels
// Don't parametrize kernels because of performance reasons
__global__ void AddKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) result[idx] = l[idx] + r[idx];
}

__global__ void SubKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) result[idx] = l[idx] - r[idx];
}

__global__ void MulKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) result[idx] = l[idx] * r[idx];
}

__global__ void DivKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) result[idx] = l[idx] / r[idx];
}

void LaunchArithmeticKernel(const ElemType* d_l, const ElemType* d_r, ElemType* d_out, size_t n, ArithmeticOp op) {
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    switch(op) {
        case ArithmeticOp::Add:
            AddKernel<<<numBlocks, blockSize>>>(d_l, d_r, d_out, n);
            break;
        case ArithmeticOp::Sub:
            SubKernel<<<numBlocks, blockSize>>>(d_l, d_r, d_out, n);
            break;
        case ArithmeticOp::Mul:
            MulKernel<<<numBlocks, blockSize>>>(d_l, d_r, d_out, n);
            break;
        case ArithmeticOp::Div:
            DivKernel<<<numBlocks, blockSize>>>(d_l, d_r, d_out, n);
            break;
    }
    cudaDeviceSynchronize();
}

// Spread kernel
template <typename T>
__global__ void SpreadKernel(const T* lhs, const T* rhs, ElemType* d_l, ElemType* d_r) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Assuming T has a static method Length() that gives the number of elements
    if (idx < T::Length()) {
        d_l[idx] = lhs->RealizeAtUnsafe(idx);
        d_r[idx] = rhs->RealizeAtUnsafe(idx);
    }
}

template <typename T>
void LaunchSpreadKernel(const T* d_lhs, const T* d_rhs, ElemType* d_l, ElemType* d_r) {
    const int blockSize = 256;
    const int numBlocks = (T::Length() + blockSize - 1) / blockSize;; // Assuming single block for simplicity
    SpreadKernel<T><<<numBlocks, blockSize>>>(d_lhs, d_rhs, d_l, d_r);
    cudaDeviceSynchronize();
}

template <typename T>
struct GPUArithmeticNaive {
    // Only arithmetic operations are done on GPU, not Spread
    static T PointwiseOpNaive(const T& lhs, const T& rhs, auto op) {
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

        LaunchArithmeticKernel(d_l, d_r, d_result, N, op);

        cudaMemcpy(result.data(), d_result, N * sizeof(ElemType), cudaMemcpyDeviceToHost);

        cudaFree(d_l);
        cudaFree(d_r);
        cudaFree(d_result);

        return T(result);
    }
    static T Add(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Add); }
    static T Sub(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Sub); }
    static T Mul(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Mul); }
    static T Div(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Div); }
};


template <typename T>
struct GPUArithmetic {
    // Spread is done on GPU
    static T PointwiseOp(const T& lhs, const T& rhs, auto op) {
        // Untested, adjustable
        const int THRESHOLD = 0;
        // For small sizes, GPU overhead is larger than computation time
        if constexpr (T::Length() <= THRESHOLD) {
            return PointwiseOpNaive(lhs, rhs, op);
        } else {
            constexpr size_t N = T::Length();
            size_t blockSizeBytes = T::SizeBytes();
            size_t flattenSizeBytes = N * sizeof(ElemType);

            std::array<ElemType, N> result;
            T *d_lhs, *d_rhs;
            ElemType *d_l, *d_r, *d_result;

            cudaMalloc(&d_lhs, blockSizeBytes);
            cudaMalloc(&d_rhs, blockSizeBytes);
            cudaMalloc(&d_l, flattenSizeBytes);
            cudaMalloc(&d_r, flattenSizeBytes);
            cudaMalloc(&d_result, flattenSizeBytes);

            cudaMemcpy(d_lhs, &lhs, blockSizeBytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_rhs, &rhs, blockSizeBytes, cudaMemcpyHostToDevice);

            LaunchSpreadKernel<T>(d_lhs, d_rhs, d_l, d_r);
            LaunchArithmeticKernel(d_l, d_r, d_result, N, op);

            cudaMemcpy(result.data(), d_result, flattenSizeBytes, cudaMemcpyDeviceToHost);

            cudaFree(d_l);
            cudaFree(d_r);
            cudaFree(d_result);

            return T(result);
        }
    }
    static T Add(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Add); }
    static T Sub(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Sub); }
    static T Mul(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Mul); }
    static T Div(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Div); }
};

#endif
