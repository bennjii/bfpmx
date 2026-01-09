#pragma once

#include "JacobiKernel.cuh"
#include "JacobiKernelBase.cuh"
#include <math.h>

#ifdef HAS_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

#include "arch/gpu/PinnedMemoryPool.h"

// Naive Jacobi 2D over a flattened N×N grid stored in an MxVector.
// Keeps boundary cells unchanged. Returns the final A buffer after `steps`.
template <typename MxVectorT>
MxVectorT Jacobi2DGPUMxVectorNaive(const MxVectorT& A,
                                   const MxVectorT& B,
                                   uint32_t N,
                                   uint32_t steps) {
    auto nvtx_range = nvtxRangeStartA("Jacobi2DGPUMxVectorNaive");
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    using BlockViewT = typename MxVectorViewT::BlockViewT;

    const size_t expected_size = static_cast<size_t>(N) * static_cast<size_t>(N);
    if (A.Size() != expected_size || B.Size() != expected_size) {
        return A;
    }

    auto& stream_pool = GetGlobalStreamPool();
    cudaStream_t compute_stream = stream_pool.Acquire();

    MxVectorViewT d_A_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(A));
    MxVectorViewT d_B_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(B));
    const uint32_t num_blocks = d_A_view.numBlocks;
    const uint32_t elems_per_block = A.NumBlockElements();
    const size_t total_elems = static_cast<size_t>(num_blocks) * elems_per_block;

    ElemType* d_A_flat = nullptr;
    ElemType* d_B_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_flat, total_elems * sizeof(ElemType)));
    CUDA_CHECK(cudaMalloc(&d_B_flat, total_elems * sizeof(ElemType)));

    LaunchBatchedSpreadKernel<BlockViewT>(
        d_A_view.blocks,
        d_B_view.blocks,
        d_A_flat,
        d_B_flat,
        num_blocks,
        elems_per_block);

    const int blockSize = 256;
    const int numBlocksLaunch = static_cast<int>((expected_size + blockSize - 1) / blockSize);

    for (uint32_t t = 0; t < steps; ++t) {
        Jacobi2DUpdateKernel<<<numBlocksLaunch, blockSize, 0, compute_stream>>>(
            d_A_flat, d_B_flat, N);
        Jacobi2DUpdateKernel<<<numBlocksLaunch, blockSize, 0, compute_stream>>>(
            d_B_flat, d_A_flat, N);
    }
    // No need to device-sync here; we synchronize the stream below before using results.
    CUDA_CHECK(cudaGetLastError());

    MxVectorViewT d_result_view = AllocateDeviceMxVectorView<BlockT>(A.Size());
    // Use requantization kernel to properly recompute scalars for each block
    LaunchBatchedPackWithRequantizeKernel<BlockViewT>(
        d_A_flat,
        d_result_view.blocks,
        d_result_view.numBlocks,
        elems_per_block,
        compute_stream);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_A_flat));
    CUDA_CHECK(cudaFree(d_B_flat));

    stream_pool.Release(compute_stream);

    FreeDeviceMxVectorView(&d_A_view);
    FreeDeviceMxVectorView(&d_B_view);

    MxVectorT result = ToHostMxVector<BlockT, MxVectorT>(d_result_view);
    FreeDeviceMxVectorView(&d_result_view);
    nvtxRangeEnd(nvtx_range);
    return result;
}

// =============================================================================
// Setup/Compute/Teardown for MxVector Naive (Spread Once)
// =============================================================================

template <typename MxVectorT>
void Jacobi2DMxVectorNaiveSetup(Jacobi2DMxVectorNaiveContext<MxVectorT>& ctx,
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
    
    // Allocate flat buffers for spread computation
    CUDA_CHECK(cudaMalloc(&ctx.d_A_flat, total_elems * sizeof(ElemType)));
    CUDA_CHECK(cudaMalloc(&ctx.d_B_flat, total_elems * sizeof(ElemType)));
    
    // Allocate result view
    ctx.d_result_view = AllocateDeviceMxVectorView<BlockT>(A.Size());
    
    ctx.initialized = true;
}

template <typename MxVectorT>
void Jacobi2DMxVectorNaiveCompute(Jacobi2DMxVectorNaiveContext<MxVectorT>& ctx, uint32_t steps) {
    if (!ctx.initialized) return;
    
    auto nvtx_range = nvtxRangeStartA("Jacobi2DMxVectorNaiveCompute");
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    using BlockViewT = typename MxVectorViewT::BlockViewT;
    
    const size_t expected_size = static_cast<size_t>(ctx.N) * static_cast<size_t>(ctx.N);
    
    // Spread MxVector data to flat buffers (once at the beginning)
    LaunchBatchedSpreadKernel<BlockViewT>(
        ctx.d_A_view.blocks,
        ctx.d_B_view.blocks,
        ctx.d_A_flat,
        ctx.d_B_flat,
        ctx.num_blocks,
        ctx.elems_per_block);
    
    // Run Jacobi iterations on flat buffers
    const int blockSize = 256;
    const int numBlocksLaunch = static_cast<int>((expected_size + blockSize - 1) / blockSize);
    
    for (uint32_t t = 0; t < steps; ++t) {
        Jacobi2DUpdateKernel<<<numBlocksLaunch, blockSize, 0, ctx.stream>>>(
            ctx.d_A_flat, ctx.d_B_flat, ctx.N);
        Jacobi2DUpdateKernel<<<numBlocksLaunch, blockSize, 0, ctx.stream>>>(
            ctx.d_B_flat, ctx.d_A_flat, ctx.N);
    }
    CUDA_CHECK(cudaGetLastError());
    
    // Requantize and pack result
    LaunchBatchedPackWithRequantizeKernel<BlockViewT>(
        ctx.d_A_flat,
        ctx.d_result_view.blocks,
        ctx.d_result_view.numBlocks,
        ctx.elems_per_block,
        ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaGetLastError());
    nvtxRangeEnd(nvtx_range);
}

template <typename MxVectorT>
MxVectorT Jacobi2DMxVectorNaiveTeardown(Jacobi2DMxVectorNaiveContext<MxVectorT>& ctx) {
    if (!ctx.initialized) {
        return MxVectorT(0);
    }
    
    using BlockT = typename MxVectorT::BlockType;
    
    // Copy result to host
    MxVectorT result = ToHostMxVector<BlockT, MxVectorT>(ctx.d_result_view, ctx.stream);
    
    // Free all device memory
    CUDA_CHECK(cudaFree(ctx.d_A_flat));
    CUDA_CHECK(cudaFree(ctx.d_B_flat));
    FreeDeviceMxVectorView(&ctx.d_A_view);
    FreeDeviceMxVectorView(&ctx.d_B_view);
    FreeDeviceMxVectorView(&ctx.d_result_view);
    
    auto& stream_pool = GetGlobalStreamPool();
    stream_pool.Release(ctx.stream);
    
    ctx.d_A_flat = nullptr;
    ctx.d_B_flat = nullptr;
    ctx.stream = nullptr;
    ctx.initialized = false;
    
    return result;
}

