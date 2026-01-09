#pragma once

#include "JacobiKernel.cuh"
#include <math.h>

// Helper function to dequantize a single element from a BlockView
// This template function needs to be available for instantiation
template <typename FloatD, typename BlockViewT>
__device__ __forceinline__ double DequantizeElement(
    const BlockViewT* blocks,
    uint32_t global_idx,
    uint32_t elems_per_block,
    uint32_t num_blocks) {
    
    if (global_idx >= num_blocks * elems_per_block) {
        return 0.0;
    }
    
    uint32_t block_idx = global_idx / elems_per_block;
    uint32_t elem_idx = global_idx % elems_per_block;
    
    if (block_idx >= num_blocks) {
        return 0.0;
    }
    
    const BlockViewT* block = &blocks[block_idx];
    
    // Match BatchedSpreadKernel's bounds checking
    if (elem_idx >= block->num_elems) {
        return 0.0;
    }
    
    // Match the spread kernel's scalar computation exactly
    double scalar = (block->scalar == 0) ? 1.0 : exp2((double)block->scalar);
    const uint8_t* data = block->data + elem_idx * block->elem_size_bytes;
    return FloatD::UnmarshalDevice(data) * scalar;
}

