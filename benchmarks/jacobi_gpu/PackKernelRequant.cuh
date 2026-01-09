#pragma once

#include "arch/gpu/common.cuh"
#include "arch/gpu/FloatReprDevice.cuh"

// Requantization pack kernel - properly recomputes scalar for each MX block
// Uses one CUDA block per MX block for efficient reduction to find max value
// This is benchmark-specific and not part of the general-purpose library
template <typename BlockViewT>
void LaunchBatchedPackWithRequantizeKernel(const ElemType* d_in,
                                           BlockViewT* d_out_blocks,
                                           uint32_t num_blocks,
                                           uint32_t elems_per_block,
                                           cudaStream_t stream = 0);

