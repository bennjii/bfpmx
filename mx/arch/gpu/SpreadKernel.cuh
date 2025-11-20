#ifndef GPU_SPREAD_KERNELS_CUH
#define GPU_SPREAD_KERNELS_CUH
#include "common.cuh"

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
#endif // GPU_SPREAD_KERNELS_CUH