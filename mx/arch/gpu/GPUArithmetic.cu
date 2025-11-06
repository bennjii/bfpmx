#include "GPUArithmetic.cuh"

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

// hosts calls this, kernels stay in a .cu file
void LaunchKernel(const ElemType* d_l, const ElemType* d_r, ElemType* d_out, size_t n, ArithmeticOp op) {
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