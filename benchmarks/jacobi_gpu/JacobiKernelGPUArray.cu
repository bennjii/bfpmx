#include "JacobiKernel.cuh"
#include <cstring>

#ifdef HAS_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

#include "arch/gpu/PinnedMemoryPool.h"

#ifndef JACOBI_KERNEL_INSTANTIATIONS_ONLY
// GPU implementation using normal CUDA arrays (f64/double)
// Uses pinned memory and compute stream to match MxVector approach
void Jacobi2DGPUArrayNaive(const ElemType* A_host,
                          const ElemType* B_host,
                          ElemType* result_host,
                          uint32_t N,
                          uint32_t steps) {
    const size_t size = static_cast<size_t>(N) * N;
    const size_t bytes = size * sizeof(ElemType);
    
    // Get compute stream from pool (matches MxVector)
    auto& stream_pool = GetGlobalStreamPool();
    cudaStream_t compute_stream = stream_pool.Acquire();
    
    // Allocate device memory
    ElemType* d_A = nullptr;
    ElemType* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    
    // Allocate pinned host memory (matches MxVector ToDeviceMxVectorView)
    void* pinned_in = nullptr;
    void* pinned_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned_in, 2 * bytes));
    CUDA_CHECK(cudaMallocHost(&pinned_out, bytes));
    
    // Copy to pinned memory then async transfer to device
    auto* pinned_A = static_cast<ElemType*>(pinned_in);
    auto* pinned_B = pinned_A + size;
    std::memcpy(pinned_A, A_host, bytes);
    std::memcpy(pinned_B, B_host, bytes);
    
    CUDA_CHECK(cudaMemcpyAsync(d_A, pinned_A, bytes, cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, pinned_B, bytes, cudaMemcpyHostToDevice, compute_stream));
    
    // Launch kernel configuration
    const int blockSize = 256;
    const int numBlocks = static_cast<int>((size + blockSize - 1) / blockSize);
    
    // Run Jacobi iterations on compute stream
    for (uint32_t t = 0; t < steps; ++t) {
        Jacobi2DUpdateKernel<<<numBlocks, blockSize, 0, compute_stream>>>(d_A, d_B, N);
        Jacobi2DUpdateKernel<<<numBlocks, blockSize, 0, compute_stream>>>(d_B, d_A, N);
    }
    
    // Copy result back to pinned memory then to user buffer
    CUDA_CHECK(cudaMemcpyAsync(pinned_out, d_A, bytes, cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaGetLastError());
    std::memcpy(result_host, pinned_out, bytes);
    
    // Free pinned and device memory
    CUDA_CHECK(cudaFreeHost(pinned_in));
    CUDA_CHECK(cudaFreeHost(pinned_out));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    
    stream_pool.Release(compute_stream);
}

// Setup: allocate device memory, pinned memory, upload data, and get stream (call once before timing)
void Jacobi2DGPUArraySetup(Jacobi2DGPUArrayContext& ctx,
                           const ElemType* A_host,
                           const ElemType* B_host,
                           uint32_t N) {
    ctx.N = N;
    ctx.bytes = static_cast<size_t>(N) * N * sizeof(ElemType);
    const size_t size = static_cast<size_t>(N) * N;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&ctx.d_A, ctx.bytes));
    CUDA_CHECK(cudaMalloc(&ctx.d_B, ctx.bytes));
    
    // Allocate pinned host memory (matches MxVector approach)
    CUDA_CHECK(cudaMallocHost(&ctx.pinned_in, 2 * ctx.bytes));
    CUDA_CHECK(cudaMallocHost(&ctx.pinned_out, ctx.bytes));
    
    // Get compute stream from pool
    auto& stream_pool = GetGlobalStreamPool();
    ctx.stream = stream_pool.Acquire();
    
    // Upload data once during setup (like MxVector does)
    auto* pinned_A = static_cast<ElemType*>(ctx.pinned_in);
    auto* pinned_B = pinned_A + size;
    std::memcpy(pinned_A, A_host, ctx.bytes);
    std::memcpy(pinned_B, B_host, ctx.bytes);
    
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_A, pinned_A, ctx.bytes, cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_B, pinned_B, ctx.bytes, cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream)); // Ensure upload completes before compute
    
    ctx.initialized = true;
}

// Compute: run Jacobi iterations on pre-allocated memory (this is what you time)
// No host-device transfers - data is already on GPU from Setup (matches MxVector approach)
void Jacobi2DGPUArrayCompute(Jacobi2DGPUArrayContext& ctx, uint32_t steps) {
    if (!ctx.initialized) return;
    
    auto nvtx_range = nvtxRangeStartA("Jacobi2DGPUArrayCompute");
    const size_t size = static_cast<size_t>(ctx.N) * ctx.N;
    
    // Launch kernel configuration
    const int blockSize = 256;
    const int numBlocks = static_cast<int>((size + blockSize - 1) / blockSize);
    
    // Run Jacobi iterations on compute stream (no transfers - data already on GPU)
    for (uint32_t t = 0; t < steps; ++t) {
        Jacobi2DUpdateKernel<<<numBlocks, blockSize, 0, ctx.stream>>>(ctx.d_A, ctx.d_B, ctx.N);
        Jacobi2DUpdateKernel<<<numBlocks, blockSize, 0, ctx.stream>>>(ctx.d_B, ctx.d_A, ctx.N);
    }
    CUDA_CHECK(cudaGetLastError());
    nvtxRangeEnd(nvtx_range);
}

// Teardown: download results, free device memory, pinned memory, and stream (call once after timing)
void Jacobi2DGPUArrayTeardown(Jacobi2DGPUArrayContext& ctx, ElemType* result_host) {
    if (!ctx.initialized) return;
    
    // Download result once during teardown (like MxVector does)
    CUDA_CHECK(cudaMemcpyAsync(ctx.pinned_out, ctx.d_A, ctx.bytes, cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaGetLastError());
    std::memcpy(result_host, ctx.pinned_out, ctx.bytes);
    
    auto& stream_pool = GetGlobalStreamPool();
    stream_pool.Release(ctx.stream);
    
    CUDA_CHECK(cudaFree(ctx.d_A));
    CUDA_CHECK(cudaFree(ctx.d_B));
    CUDA_CHECK(cudaFreeHost(ctx.pinned_in));
    CUDA_CHECK(cudaFreeHost(ctx.pinned_out));
    
    ctx.d_A = nullptr;
    ctx.d_B = nullptr;
    ctx.pinned_in = nullptr;
    ctx.pinned_out = nullptr;
    ctx.stream = nullptr;
    ctx.initialized = false;
}
#endif // JACOBI_KERNEL_INSTANTIATIONS_ONLY

