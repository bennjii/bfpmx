#include "FusedKernel.cuh"
#include "cuda_utils.h"
#include <math.h>

__device__ __forceinline__ double ComputeScalar(uint8_t scalar_bits) {
    return (scalar_bits == 0) ? 1.0 : exp2f((float)scalar_bits);
}

template <ArithmeticOp op>
__device__ __forceinline__ double PerformArithmetic(double l_val, double r_val) {
    if constexpr (op == ArithmeticOp::Add) {
        return l_val + r_val;
    } else if constexpr (op == ArithmeticOp::Sub) {
        return l_val - r_val;
    } else if constexpr (op == ArithmeticOp::Mul) {
        return l_val * r_val;
    } else if constexpr (op == ArithmeticOp::Div) {
        return l_val / r_val;
    } else {
        return l_val;
    }
}

template <typename FloatD, typename BlockViewT, ArithmeticOp op>
__global__ void FusedBlockArithmeticKernel(
    const BlockViewT* __restrict__ d_lhs_blocks,
    const BlockViewT* __restrict__ d_rhs_blocks,
    BlockViewT* __restrict__ d_out_blocks,
    uint32_t num_blocks,
    uint32_t elems_per_block,
    uint32_t blocks_per_thread_block) {
        
    const uint32_t base_block_idx = blockIdx.x * blocks_per_thread_block;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t local_block = thread_id / elems_per_block;
    const uint32_t local_elem_idx = thread_id % elems_per_block;
    
    if (local_block < blocks_per_thread_block) {
        const uint32_t block_idx = base_block_idx + local_block;
        
        if (block_idx < num_blocks && local_elem_idx < d_lhs_blocks[block_idx].num_elems) {
            const BlockViewT* l_block = &d_lhs_blocks[block_idx];
            const BlockViewT* r_block = &d_rhs_blocks[block_idx];
            BlockViewT* out_block = &d_out_blocks[block_idx];

            uint8_t l_scalar_bits = __ldg(&l_block->scalar);
            uint8_t r_scalar_bits = __ldg(&r_block->scalar);
            const double l_scalar = ComputeScalar(l_scalar_bits);
            const double r_scalar = ComputeScalar(r_scalar_bits);
            
            const uint8_t* l_data = l_block->data + local_elem_idx * l_block->elem_size_bytes;
            const uint8_t* r_data = r_block->data + local_elem_idx * r_block->elem_size_bytes;
            
            double l_val = FloatD::UnmarshalDevice(l_data) * l_scalar;
            double r_val = FloatD::UnmarshalDevice(r_data) * r_scalar;
            double result = PerformArithmetic<op>(l_val, r_val);
            
            auto packed_bytes = FloatD::MarshalDevice(result);
            #pragma unroll
            for (unsigned int b = 0; b < FloatD::SizeBytes(); ++b) {
                out_block->data[local_elem_idx * FloatD::SizeBytes() + b] = packed_bytes[b];
            }
        }
    }
}

template <typename BlockViewT, ArithmeticOp op>
void LaunchFusedBlockArithmeticKernel(
    const BlockViewT* d_lhs_blocks,
    const BlockViewT* d_rhs_blocks,
    BlockViewT* d_out_blocks,
    uint32_t num_blocks,
    uint32_t elems_per_block) {
    
    const int threadsPerBlock = 256;
    const int blocks_per_thread_block = threadsPerBlock / elems_per_block;
    const int numBlocks = (num_blocks + blocks_per_thread_block - 1) / blocks_per_thread_block;
    
    using BV = BlockViewT;
    using FloatTypeD =
        FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;
    
    FusedBlockArithmeticKernel<FloatTypeD, BlockViewT, op>
        <<<numBlocks, threadsPerBlock>>>(d_lhs_blocks, d_rhs_blocks, d_out_blocks, 
                                         num_blocks, elems_per_block, blocks_per_thread_block);
    CUDA_CHECK_KERNEL();
}

using BV431 = BlockView<4,3,1>;
using BV521 = BlockView<5,2,1>;

template
void LaunchFusedBlockArithmeticKernel<BV431, ArithmeticOp::Add>(
    const BV431*, const BV431*, BV431*, uint32_t, uint32_t);

template
void LaunchFusedBlockArithmeticKernel<BV431, ArithmeticOp::Sub>(
    const BV431*, const BV431*, BV431*, uint32_t, uint32_t);

template
void LaunchFusedBlockArithmeticKernel<BV431, ArithmeticOp::Mul>(
    const BV431*, const BV431*, BV431*, uint32_t, uint32_t);

template
void LaunchFusedBlockArithmeticKernel<BV431, ArithmeticOp::Div>(
    const BV431*, const BV431*, BV431*, uint32_t, uint32_t);

template
void LaunchFusedBlockArithmeticKernel<BV521, ArithmeticOp::Add>(
    const BV521*, const BV521*, BV521*, uint32_t, uint32_t);

template
void LaunchFusedBlockArithmeticKernel<BV521, ArithmeticOp::Sub>(
    const BV521*, const BV521*, BV521*, uint32_t, uint32_t);

template
void LaunchFusedBlockArithmeticKernel<BV521, ArithmeticOp::Mul>(
    const BV521*, const BV521*, BV521*, uint32_t, uint32_t);

template
void LaunchFusedBlockArithmeticKernel<BV521, ArithmeticOp::Div>(
    const BV521*, const BV521*, BV521*, uint32_t, uint32_t);

