#include "PackKernelRequant.cuh"
#include "arch/gpu/cuda_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Block types you want to support:
using BV431 = BlockView<4,3,1>;
using BV521 = BlockView<5,2,1>;

// Compute scalar exponent from max absolute value
// Matches MaximumFractionalQuantization::QuantizerScaleFactor:
__device__ __forceinline__ uint8_t computeScalarFromMax(double max_abs) {
    // Round to nearest integer (matches CPU lround)
    long long rounded = llrint(max_abs);
    unsigned int scaleFactorCandidate = (rounded < 0) ? 0 : static_cast<unsigned int>(rounded);
    
    if (scaleFactorCandidate == 0) {
        // When max < 0.5, lround gives 0
        // Return scalar = 0, meaning scale factor = 2^0 = 1
        return 0;
    }
    
    // Compute floor(log2(x)) using count leading zeros
    // __clz returns number of leading zeros in a 32-bit integer
    // For x > 0: 31 - __clz(x) = floor(log2(x))
    int scaleFactorInt = 31 - __clz(scaleFactorCandidate);
    
    // Clamp to valid range [0, 255]
    if (scaleFactorInt < 0) scaleFactorInt = 0;
    if (scaleFactorInt > 255) scaleFactorInt = 255;
    
    return static_cast<uint8_t>(scaleFactorInt);
}

// Simple requantization kernel: one CUDA block per MX block
// Uses shared memory with simple sequential reduction by thread 0
// Focus on correctness, not optimization
template <typename FloatD, typename BlockViewT>
__global__ void BatchedPackWithRequantizeKernel(
    const ElemType* __restrict__ d_in,
    BlockViewT* __restrict__ d_out_blocks,
    uint32_t num_blocks,
    uint32_t elems_per_block) {
    
    // Each CUDA block handles one MX block
    const uint32_t mx_block_idx = blockIdx.x;
    if (mx_block_idx >= num_blocks) return;
    
    const uint32_t elem_idx = threadIdx.x;
    BlockViewT* out_block = &d_out_blocks[mx_block_idx];
    
    // Shared memory for all values in this MX block
    extern __shared__ double s_values[];
    
    // Load value into shared memory (or 0 if out of bounds)
    double val = 0.0;
    if (elem_idx < elems_per_block && elem_idx < out_block->num_elems) {
        val = d_in[mx_block_idx * elems_per_block + elem_idx];
    }
    s_values[elem_idx] = val;
    __syncthreads();
    
    // Thread 0 does sequential max reduction (simple, correct)
    __shared__ uint8_t s_scalar;
    __shared__ double s_scale_factor;
    if (threadIdx.x == 0) {
        double max_abs = 0.0;
        for (uint32_t i = 0; i < elems_per_block && i < out_block->num_elems; i++) {
            double abs_val = fabs(s_values[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        s_scalar = computeScalarFromMax(max_abs);
        s_scale_factor = exp2((double)s_scalar);
        out_block->scalar = s_scalar;
    }
    __syncthreads();
    
    // Quantize and pack this thread's element
    if (elem_idx < elems_per_block && elem_idx < out_block->num_elems) {
        double scaled_val = s_values[elem_idx] / s_scale_factor;
        auto packed_bytes = FloatD::MarshalDevice(scaled_val);
        // Use reinterpret_cast to access array data without calling constexpr methods
        // std::array is guaranteed to be a POD-like structure, so this is safe
        const uint8_t* packed_data = reinterpret_cast<const uint8_t*>(&packed_bytes);
        
        #pragma unroll
        for (unsigned int b = 0; b < FloatD::SizeBytes(); ++b) {
            out_block->data[elem_idx * FloatD::SizeBytes() + b] = packed_data[b];
        }
    }
    
    // Ensure all writes to device memory are visible to other blocks/kernels
    __threadfence();
}

// Host wrapper for requantization pack
template <typename BlockViewT>
void LaunchBatchedPackWithRequantizeKernel(
    const ElemType* d_in,
    BlockViewT* d_out_blocks,
    uint32_t num_blocks,
    uint32_t elems_per_block,
    cudaStream_t stream) {
    
    using BV = BlockViewT;
    using FloatTypeD = FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;
    
    // Launch one CUDA block per MX block, with elems_per_block threads
    // Shared memory size = elems_per_block * sizeof(double)
    size_t shared_mem_size = elems_per_block * sizeof(double);
    
    BatchedPackWithRequantizeKernel<FloatTypeD, BlockViewT>
        <<<num_blocks, elems_per_block, shared_mem_size, stream>>>(
            d_in, d_out_blocks, num_blocks, elems_per_block);
    
    // Check for kernel launch errors (but don't sync here - caller handles sync)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Pack kernel launch error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// Explicit instantiations for requantization kernel
template void LaunchBatchedPackWithRequantizeKernel<BV431>(
    const ElemType*, BV431*, uint32_t, uint32_t, cudaStream_t);

template void LaunchBatchedPackWithRequantizeKernel<BV521>(
    const ElemType*, BV521*, uint32_t, uint32_t, cudaStream_t);

