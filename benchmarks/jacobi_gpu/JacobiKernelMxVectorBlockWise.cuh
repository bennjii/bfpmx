#pragma once

#include "JacobiKernel.cuh"
#include "JacobiKernelBase.cuh"
#include <math.h>

#ifdef HAS_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

#include "arch/gpu/PinnedMemoryPool.h"

// Block-wise tiled Jacobi kernel with shared memory optimization
template <typename FloatD, typename BlockViewT>
__global__ void Jacobi2DBlockWiseTiledKernel(
    const BlockViewT* __restrict__ d_source_blocks,
    BlockViewT* __restrict__ d_dest_blocks,
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

    // Destination block/element indices for this cell
    const uint32_t block_idx = linear_idx / elems_per_block;
    const uint32_t elem_idx = linear_idx % elems_per_block;
    if (block_idx >= num_blocks) return;

    const BlockViewT* src_block = &d_source_blocks[block_idx];
    BlockViewT* dst_block = &d_dest_blocks[block_idx];

    // Copy scalar (bits) once per destination block (idempotent)
    if (elem_idx == 0) {
        dst_block->scalar = src_block->scalar;
    }

    // Preserve boundary cells exactly by copying quantized bytes directly.
    if (row == 0 || col == 0 || row >= static_cast<int>(N) - 1 || col >= static_cast<int>(N) - 1) {
        if (elem_idx < dst_block->num_elems) {
            const uint8_t* src_data = src_block->data + elem_idx * src_block->elem_size_bytes;
            uint8_t* dst_data = dst_block->data + elem_idx * dst_block->elem_size_bytes;
            #pragma unroll
            for (unsigned int b = 0; b < FloatD::SizeBytes(); ++b) {
                dst_data[b] = src_data[b];
            }
        }
        return;
    }

    // 5-point stencil from shared memory
    const double val_center = s_tile[ty + 1][tx + 1];
    const double val_left   = s_tile[ty + 1][tx + 0];
    const double val_right  = s_tile[ty + 1][tx + 2];
    const double val_up     = s_tile[ty + 0][tx + 1];
    const double val_down   = s_tile[ty + 2][tx + 1];

    const double result = 0.2 * (val_center + val_left + val_right + val_up + val_down);

    // Pack using source scalar (same convention as DequantizeElement)
    const double scalar = (src_block->scalar == 0) ? 1.0 : exp2((double)src_block->scalar);
    const double scaled_result = result / scalar;
    const auto packed_bytes = FloatD::MarshalDevice(scaled_result);
    // Use reinterpret_cast to access array data without calling constexpr methods
    // std::array is guaranteed to be a POD-like structure, so this is safe
    const uint8_t* packed_data = reinterpret_cast<const uint8_t*>(&packed_bytes);

    if (elem_idx < dst_block->num_elems) {
        uint8_t* dst_data = dst_block->data + elem_idx * dst_block->elem_size_bytes;
        #pragma unroll
        for (unsigned int b = 0; b < FloatD::SizeBytes(); ++b) {
            dst_data[b] = packed_data[b];
        }
    }
}

// Block-wise Jacobi 2D without requantization (keeps source scalar)
// Uses fused tiled kernel that packs directly using source block scalar.
// Lower overhead but scalars don't adapt to changing data distribution.
template <typename MxVectorT>
MxVectorT Jacobi2DGPUMxVectorBlockWise(const MxVectorT& A,
                                       const MxVectorT& B,
                                       uint32_t N,
                                       uint32_t steps) {
    auto nvtx_range = nvtxRangeStartA("Jacobi2DGPUMxVectorBlockWise");
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

    using BV = BlockViewT;
    using FloatTypeD =
        FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;

    // Run Jacobi iterations with fused update+pack.
    // Each step does 2 Jacobi updates (A->B, B->A), matching the naive implementation.
    for (uint32_t t = 0; t < steps; ++t) {
        // First half-step: A -> B (tiled shared-memory update, pack using A's block scalar)
        {
            constexpr int BDimX = 32;
            constexpr int BDimY = 16;
            dim3 block(BDimX, BDimY);
            dim3 grid((N + BDimX - 1) / BDimX, (N + BDimY - 1) / BDimY);
            Jacobi2DBlockWiseTiledKernel<FloatTypeD, BlockViewT>
                <<<grid, block, 0, compute_stream>>>(
                    d_A_view.blocks, d_B_view.blocks, N, elems_per_block, num_blocks);
        }

        // Second half-step: B -> A (tiled shared-memory update, pack using B's block scalar)
        {
            constexpr int BDimX = 32;
            constexpr int BDimY = 16;
            dim3 block(BDimX, BDimY);
            dim3 grid((N + BDimX - 1) / BDimX, (N + BDimY - 1) / BDimY);
            Jacobi2DBlockWiseTiledKernel<FloatTypeD, BlockViewT>
                <<<grid, block, 0, compute_stream>>>(
                    d_B_view.blocks, d_A_view.blocks, N, elems_per_block, num_blocks);
        }
    }
    // Synchronize to check for kernel errors before data transfer
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaGetLastError());
    
    // ToHostMxVector will synchronize again before reading (redundant but safe)
    MxVectorT result = ToHostMxVector<BlockT, MxVectorT>(d_A_view, compute_stream);
    stream_pool.Release(compute_stream);
    FreeDeviceMxVectorView(&d_A_view);
    FreeDeviceMxVectorView(&d_B_view);
    nvtxRangeEnd(nvtx_range);
    return result;
}

// =============================================================================
// Setup/Compute/Teardown for MxVector BlockWise (No Requant)
// =============================================================================

template <typename MxVectorT>
void Jacobi2DMxVectorBlockWiseSetup(Jacobi2DMxVectorBlockWiseContext<MxVectorT>& ctx,
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
    
    ctx.initialized = true;
}

template <typename MxVectorT>
void Jacobi2DMxVectorBlockWiseCompute(Jacobi2DMxVectorBlockWiseContext<MxVectorT>& ctx, uint32_t steps) {
    if (!ctx.initialized) return;
    
    auto nvtx_range = nvtxRangeStartA("Jacobi2DMxVectorBlockWiseCompute");
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
        // First half-step: A -> B
        Jacobi2DBlockWiseTiledKernel<FloatTypeD, BlockViewT>
            <<<grid, block, 0, ctx.stream>>>(
                ctx.d_A_view.blocks, ctx.d_B_view.blocks, ctx.N, ctx.elems_per_block, ctx.num_blocks);
        
        // Second half-step: B -> A
        Jacobi2DBlockWiseTiledKernel<FloatTypeD, BlockViewT>
            <<<grid, block, 0, ctx.stream>>>(
                ctx.d_B_view.blocks, ctx.d_A_view.blocks, ctx.N, ctx.elems_per_block, ctx.num_blocks);
    }
    
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaGetLastError());
    nvtxRangeEnd(nvtx_range);
}

template <typename MxVectorT>
MxVectorT Jacobi2DMxVectorBlockWiseTeardown(Jacobi2DMxVectorBlockWiseContext<MxVectorT>& ctx) {
    if (!ctx.initialized) {
        return MxVectorT(0);
    }
    
    using BlockT = typename MxVectorT::BlockType;
    
    // Copy result (A view) to host
    MxVectorT result = ToHostMxVector<BlockT, MxVectorT>(ctx.d_A_view, ctx.stream);
    
    // Free device views
    FreeDeviceMxVectorView(&ctx.d_A_view);
    FreeDeviceMxVectorView(&ctx.d_B_view);
    
    auto& stream_pool = GetGlobalStreamPool();
    stream_pool.Release(ctx.stream);
    
    ctx.stream = nullptr;
    ctx.initialized = false;
    
    return result;
}

