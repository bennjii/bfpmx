#ifndef GPU_FUSED_KERNEL_CUH
#define GPU_FUSED_KERNEL_CUH

#include "common.cuh"
#include "FloatReprDevice.cuh"

template <typename FloatD, typename BlockViewT, ArithmeticOp op>
__global__ void FusedBlockArithmeticKernel(
    const BlockViewT* __restrict__ d_lhs_blocks,
    const BlockViewT* __restrict__ d_rhs_blocks,
    BlockViewT* __restrict__ d_out_blocks,
    uint32_t num_blocks,
    uint32_t elems_per_block,
    uint32_t blocks_per_thread_block);

template <typename BlockViewT, ArithmeticOp op>
void LaunchFusedBlockArithmeticKernel(
    const BlockViewT* d_lhs_blocks,
    const BlockViewT* d_rhs_blocks,
    BlockViewT* d_out_blocks,
    uint32_t num_blocks,
    uint32_t elems_per_block);

#endif // GPU_FUSED_KERNEL_CUH

