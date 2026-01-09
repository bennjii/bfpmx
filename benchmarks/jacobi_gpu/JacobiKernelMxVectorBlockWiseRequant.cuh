#pragma once

#include "JacobiKernel.cuh"
#include "JacobiKernelBase.cuh"
#include <math.h>

#ifdef HAS_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

#include "arch/gpu/PinnedMemoryPool.h"

// Tiled Jacobi kernel that writes results to a flat f64 buffer (for requantization)
// Similar to Jacobi2DBlockWiseTiledKernel but outputs to ElemType* instead of BlockViewT*
template <typename FloatD, typename BlockViewT>
__global__ void Jacobi2DBlockWiseTiledToFlatKernel(
    const BlockViewT* __restrict__ d_source_blocks,
    ElemType* __restrict__ d_out_flat,
    uint32_t N,
    uint32_t elems_per_block,
    uint32_t num_blocks) 
{
    // Fixed tiling parameters (must match launch configuration).
    // Shared memory contains a halo of 1 in each direction.
    constexpr int BDimX = 32;
    constexpr int BDimY = 16;

    __shared__ double s_tile[BDimY + 2][BDimX + 2];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    const int col = static_cast<int>(blockIdx.x) * BDimX + tx;
    const int row = static_cast<int>(blockIdx.y) * BDimY + ty;

    const bool is_valid = (col >= 0 && row >= 0 && col < static_cast<int>(N) && row < static_cast<int>(N));
    const uint32_t linear_idx = static_cast<uint32_t>(row) * N + static_cast<uint32_t>(col);

    // Center load
    if (is_valid) {
        s_tile[ty + 1][tx + 1] = DequantizeElement<FloatD>(d_source_blocks, linear_idx, elems_per_block, num_blocks);
    } else {
        s_tile[ty + 1][tx + 1] = 0.0;
    }

    // Halo loads for cross-neighbors only (no corners needed for 5-point stencil).
    if (tx == 0) {
        if (row < static_cast<int>(N) && col > 0) {
            s_tile[ty + 1][0] = DequantizeElement<FloatD>(d_source_blocks, linear_idx - 1, elems_per_block, num_blocks);
        } else {
            s_tile[ty + 1][0] = 0.0;
        }
    }
    if (tx == BDimX - 1) {
        if (row < static_cast<int>(N) && col < static_cast<int>(N) - 1) {
            s_tile[ty + 1][BDimX + 1] = DequantizeElement<FloatD>(d_source_blocks, linear_idx + 1, elems_per_block, num_blocks);
        } else {
            s_tile[ty + 1][BDimX + 1] = 0.0;
        }
    }
    if (ty == 0) {
        if (col < static_cast<int>(N) && row > 0) {
            s_tile[0][tx + 1] = DequantizeElement<FloatD>(d_source_blocks, linear_idx - N, elems_per_block, num_blocks);
        } else {
            s_tile[0][tx + 1] = 0.0;
        }
    }
    if (ty == BDimY - 1) {
        if (col < static_cast<int>(N) && row < static_cast<int>(N) - 1) {
            s_tile[BDimY + 1][tx + 1] = DequantizeElement<FloatD>(d_source_blocks, linear_idx + N, elems_per_block, num_blocks);
        } else {
            s_tile[BDimY + 1][tx + 1] = 0.0;
        }
    }

    __syncthreads();

    // After the barrier it's safe to early-return.
    if (!is_valid) return;

    // Preserve boundary cells exactly by copying from source
    if (row == 0 || col == 0 || row >= static_cast<int>(N) - 1 || col >= static_cast<int>(N) - 1) {
        d_out_flat[linear_idx] = s_tile[ty + 1][tx + 1];
        return;
    }

    // 5-point stencil from shared memory
    const double val_center = s_tile[ty + 1][tx + 1];
    const double val_left   = s_tile[ty + 1][tx + 0];
    const double val_right  = s_tile[ty + 1][tx + 2];
    const double val_up     = s_tile[ty + 0][tx + 1];
    const double val_down   = s_tile[ty + 2][tx + 1];

    const double result = 0.2 * (val_center + val_left + val_right + val_up + val_down);
    d_out_flat[linear_idx] = result;
}

// Block-wise Jacobi 2D with requantization after each full iteration.
// Uses a two-phase approach per iteration:
// 1. Compute both half-steps (A->temp, temp->A) in full precision
// 2. Requantize A blocks with properly computed scalars
// Higher overhead but scalars adapt to changing data distribution.
template <typename MxVectorT>
MxVectorT Jacobi2DGPUMxVectorBlockWiseRequant(const MxVectorT& A,
                                               const MxVectorT& B,
                                               uint32_t N,
                                               uint32_t steps) {
    auto nvtx_range = nvtxRangeStartA("Jacobi2DGPUMxVectorBlockWiseRequant");
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    using BlockViewT = typename MxVectorViewT::BlockViewT;

    const size_t expected_size = static_cast<size_t>(N) * static_cast<size_t>(N);
    if (A.Size() != expected_size || B.Size() != expected_size) {
        return A;
    }

    auto& stream_pool = GetGlobalStreamPool();
    cudaStream_t compute_stream = stream_pool.Acquire();

    // Use compute_stream for data transfers to avoid extra synchronizations
    MxVectorViewT d_A_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(A), compute_stream);
    MxVectorViewT d_B_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(B), compute_stream);

    const uint32_t num_blocks = d_A_view.numBlocks;
    const uint32_t elems_per_block = A.NumBlockElements();
    const size_t total_elems = static_cast<size_t>(num_blocks) * elems_per_block;

    using BV = BlockViewT;
    using FloatTypeD =
        FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;

    // Allocate temporary f64 buffers for intermediate results
    ElemType* d_temp_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_flat, total_elems * sizeof(ElemType)));

    constexpr int BDimX = 32;
    constexpr int BDimY = 16;
    dim3 block(BDimX, BDimY);
    dim3 grid((N + BDimX - 1) / BDimX, (N + BDimY - 1) / BDimY);

    // Run Jacobi iterations with requantization after each full iteration.
    for (uint32_t t = 0; t < steps; ++t) {
        // First half-step: A -> temp (compute to flat buffer)
        Jacobi2DBlockWiseTiledToFlatKernel<FloatTypeD, BlockViewT>
            <<<grid, block, 0, compute_stream>>>(
                d_A_view.blocks, d_temp_flat, N, elems_per_block, num_blocks);

        // Requantize temp -> B blocks
        LaunchBatchedPackWithRequantizeKernel<BlockViewT>(
            d_temp_flat, d_B_view.blocks, num_blocks, elems_per_block, compute_stream);

        // Second half-step: B -> temp (compute to flat buffer)
        Jacobi2DBlockWiseTiledToFlatKernel<FloatTypeD, BlockViewT>
            <<<grid, block, 0, compute_stream>>>(
                d_B_view.blocks, d_temp_flat, N, elems_per_block, num_blocks);

        // Requantize temp -> A blocks
        LaunchBatchedPackWithRequantizeKernel<BlockViewT>(
            d_temp_flat, d_A_view.blocks, num_blocks, elems_per_block, compute_stream);
    }

    // Synchronize to check for kernel errors before data transfer
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffer
    CUDA_CHECK(cudaFree(d_temp_flat));
    
    // ToHostMxVector will synchronize again before reading (redundant but safe)
    MxVectorT result = ToHostMxVector<BlockT, MxVectorT>(d_A_view, compute_stream);
    stream_pool.Release(compute_stream);
    FreeDeviceMxVectorView(&d_A_view);
    FreeDeviceMxVectorView(&d_B_view);
    nvtxRangeEnd(nvtx_range);
    return result;
}

// =============================================================================
// Setup/Compute/Teardown for MxVector BlockWise with Requantization
// =============================================================================

template <typename MxVectorT>
void Jacobi2DMxVectorBlockWiseRequantSetup(Jacobi2DMxVectorBlockWiseRequantContext<MxVectorT>& ctx,
                                            const MxVectorT& A, const MxVectorT& B, uint32_t N) {
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    
    const size_t expected_size = static_cast<size_t>(N) * static_cast<size_t>(N);
    if (A.Size() != expected_size || B.Size() != expected_size) {
        ctx.initialized = false;
        return;
    }
    
    ctx.N = N;
    ctx.elems_per_block = A.NumBlockElements();
    
    auto& stream_pool = GetGlobalStreamPool();
    ctx.stream = stream_pool.Acquire();
    
    // Allocate device views for A and B
    ctx.d_A_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(A), ctx.stream);
    ctx.d_B_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(B), ctx.stream);
    ctx.num_blocks = ctx.d_A_view.numBlocks;
    
    const size_t total_elems = static_cast<size_t>(ctx.num_blocks) * ctx.elems_per_block;
    
    // Allocate temp buffer for requantization
    CUDA_CHECK(cudaMalloc(&ctx.d_temp_flat, total_elems * sizeof(ElemType)));
    
    ctx.initialized = true;
}

template <typename MxVectorT>
void Jacobi2DMxVectorBlockWiseRequantCompute(Jacobi2DMxVectorBlockWiseRequantContext<MxVectorT>& ctx, uint32_t steps) {
    if (!ctx.initialized) return;
    
    auto nvtx_range = nvtxRangeStartA("Jacobi2DMxVectorBlockWiseRequantCompute");
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    using BlockViewT = typename MxVectorViewT::BlockViewT;
    using BV = BlockViewT;
    using FloatTypeD = FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;
    
    constexpr int BDimX = 32;
    constexpr int BDimY = 16;
    dim3 block(BDimX, BDimY);
    dim3 grid((ctx.N + BDimX - 1) / BDimX, (ctx.N + BDimY - 1) / BDimY);
    
    for (uint32_t t = 0; t < steps; ++t) {
        // First half-step: A -> temp (compute to flat buffer)
        Jacobi2DBlockWiseTiledToFlatKernel<FloatTypeD, BlockViewT>
            <<<grid, block, 0, ctx.stream>>>(
                ctx.d_A_view.blocks, ctx.d_temp_flat, ctx.N, ctx.elems_per_block, ctx.num_blocks);
        
        // Requantize temp -> B blocks
        LaunchBatchedPackWithRequantizeKernel<BlockViewT>(
            ctx.d_temp_flat, ctx.d_B_view.blocks, ctx.num_blocks, ctx.elems_per_block, ctx.stream);
        
        // Second half-step: B -> temp (compute to flat buffer)
        Jacobi2DBlockWiseTiledToFlatKernel<FloatTypeD, BlockViewT>
            <<<grid, block, 0, ctx.stream>>>(
                ctx.d_B_view.blocks, ctx.d_temp_flat, ctx.N, ctx.elems_per_block, ctx.num_blocks);
        
        // Requantize temp -> A blocks
        LaunchBatchedPackWithRequantizeKernel<BlockViewT>(
            ctx.d_temp_flat, ctx.d_A_view.blocks, ctx.num_blocks, ctx.elems_per_block, ctx.stream);
    }
    
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaGetLastError());
    nvtxRangeEnd(nvtx_range);
}

template <typename MxVectorT>
MxVectorT Jacobi2DMxVectorBlockWiseRequantTeardown(Jacobi2DMxVectorBlockWiseRequantContext<MxVectorT>& ctx) {
    if (!ctx.initialized) {
        return MxVectorT(0);
    }
    
    using BlockT = typename MxVectorT::BlockType;
    
    // Copy result (A view) to host
    MxVectorT result = ToHostMxVector<BlockT, MxVectorT>(ctx.d_A_view, ctx.stream);
    
    // Free temp buffer and device views
    CUDA_CHECK(cudaFree(ctx.d_temp_flat));
    FreeDeviceMxVectorView(&ctx.d_A_view);
    FreeDeviceMxVectorView(&ctx.d_B_view);
    
    auto& stream_pool = GetGlobalStreamPool();
    stream_pool.Release(ctx.stream);
    
    ctx.d_temp_flat = nullptr;
    ctx.stream = nullptr;
    ctx.initialized = false;
    
    return result;
}

