#include "SpreadKernel.cuh"
#include "cuda_utils.h"
#include <math.h>

template <typename FloatD, typename BlockViewT>
__global__ void SpreadKernel(const BlockViewT* left_view,
                             const BlockViewT* right_view,
                             ElemType* d_l,
                             ElemType* d_r) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < left_view->num_elems) {
        double l_scalar = (left_view->scalar == 0) ? 1.0 : exp2f((float)left_view->scalar);
        double r_scalar = (right_view->scalar == 0) ? 1.0 : exp2f((float)right_view->scalar);
        d_l[idx] = FloatD::UnmarshalDevice(left_view->data + idx * left_view->elem_size_bytes) * l_scalar;
        d_r[idx] = FloatD::UnmarshalDevice(right_view->data + idx * right_view->elem_size_bytes) * r_scalar;
    }
}

template <typename BlockViewT>
void LaunchSpreadKernel(const BlockViewT* d_lhs,
                        const BlockViewT* d_rhs,
                        ElemType* d_l,
                        ElemType* d_r) {
    const int blockSize = 256;
    const int numBlocks = (d_lhs->num_elems + blockSize - 1) / blockSize;

    using BV = BlockViewT;
    using FloatTypeD =
        FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;

    SpreadKernel<FloatTypeD, BlockViewT>
        <<<numBlocks, blockSize>>>(d_lhs, d_rhs, d_l, d_r);
    // This call included cudaDeviceSynchronize()
    CUDA_CHECK_KERNEL();
}


template <typename FloatD, typename BlockViewT>
__global__ void BatchedSpreadKernel(const BlockViewT* d_lhs_blocks,
                                    const BlockViewT* d_rhs_blocks,
                                    ElemType* d_l,
                                    ElemType* d_r,
                                    uint32_t num_blocks,
                                    uint32_t elems_per_block) {
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_elems = num_blocks * elems_per_block;
    
    if (global_idx < total_elems) {
        uint32_t block_idx = global_idx / elems_per_block;
        uint32_t elem_idx = global_idx % elems_per_block;
        
        const BlockViewT* l_block = &d_lhs_blocks[block_idx];
        const BlockViewT* r_block = &d_rhs_blocks[block_idx];
        
        if (elem_idx < l_block->num_elems) {
            double l_scalar = (l_block->scalar == 0) ? 1.0 : exp2f((float)l_block->scalar);
            double r_scalar = (r_block->scalar == 0) ? 1.0 : exp2f((float)r_block->scalar);
            d_l[global_idx] = FloatD::UnmarshalDevice(l_block->data + elem_idx * l_block->elem_size_bytes) * l_scalar;
            d_r[global_idx] = FloatD::UnmarshalDevice(r_block->data + elem_idx * r_block->elem_size_bytes) * r_scalar;
        } else {
            d_l[global_idx] = 0.0;
            d_r[global_idx] = 0.0;
        }
    }
}

template <typename BlockViewT>
void LaunchBatchedSpreadKernel(const BlockViewT* d_lhs_blocks,
                               const BlockViewT* d_rhs_blocks,
                               ElemType* d_l,
                               ElemType* d_r,
                               uint32_t num_blocks,
                               uint32_t elems_per_block) {
    const int blockSize = 256;
    int total_elems = num_blocks * elems_per_block;
    const int numBlocks = (total_elems + blockSize - 1) / blockSize;
    
    using BV = BlockViewT;
    using FloatTypeD =
        FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;
    
    BatchedSpreadKernel<FloatTypeD, BlockViewT>
        <<<numBlocks, blockSize>>>(d_lhs_blocks, d_rhs_blocks, d_l, d_r, num_blocks, elems_per_block);
    CUDA_CHECK_KERNEL();
}

using BV431 = BlockView<4,3,1>;
using BV521 = BlockView<5,2,1>;

template __global__
void SpreadKernel<FloatReprDevice<4,3,1>, BV431>(
    const BV431*, const BV431*, ElemType*, ElemType*);

template __global__
void SpreadKernel<FloatReprDevice<5,2,1>, BV521>(
    const BV521*, const BV521*, ElemType*, ElemType*);

template
void LaunchSpreadKernel<BV431>(
    const BV431*, const BV431*, ElemType*, ElemType*);

template
void LaunchSpreadKernel<BV521>(
    const BV521*, const BV521*, ElemType*, ElemType*);

template
void LaunchBatchedSpreadKernel<BV431>(
    const BV431*, const BV431*, ElemType*, ElemType*, uint32_t, uint32_t);

template
void LaunchBatchedSpreadKernel<BV521>(
    const BV521*, const BV521*, ElemType*, ElemType*, uint32_t, uint32_t);

