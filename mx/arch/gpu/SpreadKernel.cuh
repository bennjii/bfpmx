#ifndef GPU_SPREAD_KERNELS_CUH
#define GPU_SPREAD_KERNELS_CUH

#include "common.cuh"
#include "FloatReprDevice.cuh"

template <typename FloatD, typename BlockT>
__global__ void SpreadKernel(const BlockT* left_view,
                             const BlockT* right_view,
                             ElemType* d_l,
                             ElemType* d_r);

template <typename BlockT>
void LaunchSpreadKernel(const BlockT* d_lhs,
                        const BlockT* d_rhs,
                        ElemType* d_l,
                        ElemType* d_r);

template <typename FloatD, typename BlockT>
__global__ void BatchedSpreadKernel(const BlockT* d_lhs_blocks,
                                    const BlockT* d_rhs_blocks,
                                    ElemType* d_l,
                                    ElemType* d_r,
                                    uint32_t num_blocks,
                                    uint32_t elems_per_block);

template <typename BlockT>
void LaunchBatchedSpreadKernel(const BlockT* d_lhs_blocks,
                               const BlockT* d_rhs_blocks,
                               ElemType* d_l,
                               ElemType* d_r,
                               uint32_t num_blocks,
                               uint32_t elems_per_block);

#endif