#include "cuda_utils.h"
#include "PackKernel.cuh"  // for ElemType

// Block types you want to support:
using BV431 = BlockView<4,3,1>;
using BV521 = BlockView<5,2,1>;

template <typename FloatD, typename BlockViewT>
__global__ void BatchedPackKernel(const ElemType* d_in,
                                 BlockViewT* d_out_blocks,
                                 uint32_t num_blocks,
                                 uint32_t elems_per_block) {
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_elems = num_blocks * elems_per_block;
    
    if (global_idx < total_elems) {
        uint32_t block_idx = global_idx / elems_per_block;
        uint32_t elem_idx = global_idx % elems_per_block;
        
        BlockViewT* out_block = &d_out_blocks[block_idx];
        
        if (elem_idx < out_block->num_elems) {
            double val = d_in[global_idx];
            auto packed_bytes = FloatD::MarshalDevice(val);
            
            #pragma unroll
            for (unsigned int b = 0; b < FloatD::SizeBytes(); ++b) {
                out_block->data[elem_idx * FloatD::SizeBytes() + b] = packed_bytes[b];
            }
        }
    }
}

template <typename BlockViewT>
void LaunchBatchedPackKernel(const ElemType* d_in,
                            BlockViewT* d_out_blocks,
                            uint32_t num_blocks,
                            uint32_t elems_per_block) {
    const int blockSize = 256;
    int total_elems = num_blocks * elems_per_block;
    const int numBlocks = (total_elems + blockSize - 1) / blockSize;
    
    using BV = BlockViewT;
    using FloatTypeD =
        FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;
    
    BatchedPackKernel<FloatTypeD, BlockViewT>
        <<<numBlocks, blockSize>>>(d_in, d_out_blocks, num_blocks, elems_per_block);
    CUDA_CHECK_KERNEL();
}

template
void LaunchBatchedPackKernel<BV431>(
    const ElemType*, BV431*, uint32_t, uint32_t);

template
void LaunchBatchedPackKernel<BV521>(
    const ElemType*, BV521*, uint32_t, uint32_t);