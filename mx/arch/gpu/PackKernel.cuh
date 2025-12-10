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

template <typename FloatD, typename BlockViewT>
__global__ void BatchedPackKernel(const ElemType* d_in,
                                  BlockViewT* d_out_blocks,
                                  uint32_t num_blocks,
                                  uint32_t elems_per_block);

template <typename BlockViewT>
void LaunchBatchedPackKernel(const ElemType* d_in,
                             BlockViewT* d_out_blocks,
                             uint32_t num_blocks,
                             uint32_t elems_per_block);

#endif