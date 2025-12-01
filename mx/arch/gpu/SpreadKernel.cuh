#ifndef GPU_SPREAD_KERNELS_CUH
#define GPU_SPREAD_KERNELS_CUH

#include "common.cuh"
#include "arch/gpu/FloatReprMap.cuh"

template <typename FloatD>
__global__ void SpreadKernel(const BlockView* left_view,
                             const BlockView* right_view,
                             ElemType* d_l,
                             ElemType* d_r);

template <typename BlockT>
void LaunchSpreadKernel(const BlockView* d_lhs,
                        const BlockView* d_rhs,
                        ElemType* d_l,
                        ElemType* d_r);

#endif