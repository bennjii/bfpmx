#ifndef GPU_SPREAD_KERNELS_CUH
#define GPU_SPREAD_KERNELS_CUH
#include "common.cuh"
#include "SpreadKernel.cuh"
#include "FloatReprMap.cuh"

template <typename FloatD>
__global__ void SpreadKernel(const BlockView* left_view, const BlockView* right_view, ElemType* d_l, ElemType* d_r) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Assuming T has a static method Length() that gives the number of elements
    if (idx < left_view->num_elems) {
        d_l[idx] = FloatD::UnmarshalDevice(left_view->data[idx * left_view->elem_size_bytes]);
        d_r[idx] = FloatD::UnmarshalDevice(right_view->data[idx * right_view->elem_size_bytes]);
    }
}

template <typename BlockT>
void LaunchSpreadKernel(const BlockView* d_lhs, const BlockView* d_rhs, ElemType* d_l, ElemType* d_r) {
    const int blockSize = 256;
    const int numBlocks = (BlockT::Length() + blockSize - 1) / blockSize;; // Assuming single block for simplicity
    
    using FloatTypeH = typename BlockT::FloatType;
    using FloatTypeD = typename DeviceFloatReprOf<FloatTypeH>::type;
    SpreadKernel<FloatTypeD><<<numBlocks, blockSize>>>(d_lhs, d_rhs, d_l, d_r);
    cudaDeviceSynchronize();
}
#endif // GPU_SPREAD_KERNELS_CUH