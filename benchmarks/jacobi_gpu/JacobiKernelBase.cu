#include "JacobiKernelBase.cuh"

// Basic Jacobi 2D update kernel - used by all implementations
#ifndef JACOBI_KERNEL_INSTANTIATIONS_ONLY
__global__ void Jacobi2DUpdateKernel(const ElemType* A,
                                     ElemType* B,
                                     uint32_t N) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = N * N;
    if (idx >= total) return;

    const uint32_t i = idx / N;
    const uint32_t j = idx % N;

    if (i == 0 || j == 0 || i >= N - 1 || j >= N - 1) {
        B[idx] = A[idx];
        return;
    }

    const uint32_t center = idx;
    const uint32_t left   = i * N + (j - 1);
    const uint32_t right  = i * N + (j + 1);
    const uint32_t up     = (i - 1) * N + j;
    const uint32_t down   = (i + 1) * N + j;

    B[idx] = static_cast<ElemType>(
        0.2 * (A[center] + A[left] + A[right] + A[up] + A[down])
    );
}
#endif // JACOBI_KERNEL_INSTANTIATIONS_ONLY
