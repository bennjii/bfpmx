#include "GPUArithmetic.cuh"

__global__ void AddKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) result[idx] = l[idx] + r[idx];
}

// hosts calls this, kernels stay in a .cu file
void launchAddKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n) {
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    AddKernel<<<numBlocks, blockSize>>>(l, r, result, n);
    cudaDeviceSynchronize();
}