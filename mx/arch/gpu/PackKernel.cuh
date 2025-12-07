#ifndef GPU_PACK_KERNELS_CUH
#define GPU_PACK_KERNELS_CUH

#include "common.cuh"
#include "FloatReprDevice.cuh"

template <typename FloatD, typename BlockViewT>
__global__ void PackKernel(const ElemType* d_in,
                           BlockViewT* out_block);

template <typename BlockViewT>
void LaunchPackKernel(const ElemType* d_in,
                      BlockViewT* out_block);

#endif