#ifndef GPU_SPREAD_KERNELS_CUH
#define GPU_SPREAD_KERNELS_CUH

#include "common.cuh"
#include "arch/gpu/FloatReprMap.cuh"

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

#endif