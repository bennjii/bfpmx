#include "JacobiKernel.cuh"

#ifndef JACOBI_KERNEL_INSTANTIATIONS_ONLY
__global__ void Jacobi2DUpdateKernel(const ElemType* __restrict__ A,
                                     ElemType* __restrict__ B,
                                     uint32_t N) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = N * N;
    if (idx >= total) return;

    const uint32_t i = idx / N;
    const uint32_t j = idx % N;

    // Keep boundary cells unchanged
    if (i == 0 || j == 0 || i >= N - 1 || j >= N - 1) {
        B[idx] = A[idx];
        return;
    }

    const uint32_t center = idx;
    const uint32_t left   = i * N + (j - 1);
    const uint32_t right  = i * N + (j + 1);
    const uint32_t up     = (i - 1) * N + j;
    const uint32_t down   = (i + 1) * N + j;

    B[idx] = static_cast<ElemType>(
        0.2 * (A[center] + A[left] + A[right] + A[up] + A[down])
    );
}
#endif // JACOBI_KERNEL_INSTANTIATIONS_ONLY

// Naive Jacobi 2D over a flattened N×N grid stored in an MxVector.
// Keeps boundary cells unchanged. Returns the final A buffer after `steps`.
template <typename MxVectorT>
MxVectorT Jacobi2DGPUMxVectorNaive(const MxVectorT& A,
                                   const MxVectorT& B,
                                   uint32_t N,
                                   uint32_t steps) {
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    using BlockViewT = typename MxVectorViewT::BlockViewT;

    const size_t expected_size = static_cast<size_t>(N) * static_cast<size_t>(N);
    if (A.Size() != expected_size || B.Size() != expected_size) {
        return A;
    }

    auto& stream_pool = GetGlobalStreamPool();
    cudaStream_t compute_stream = stream_pool.Acquire();

    MxVectorViewT d_A_view, d_B_view;
    bool need_free_A = true;
    bool need_free_B = true;

#ifdef HAS_CUDA
    if (A.getDataLocation() == mx::vector::DataLocation::GPU_ONLY ||
        A.getDataLocation() == mx::vector::DataLocation::BOTH) {
        d_A_view = A.getGPUView();
        need_free_A = false;
    } else {
        d_A_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(A));
    }

    if (B.getDataLocation() == mx::vector::DataLocation::GPU_ONLY ||
        B.getDataLocation() == mx::vector::DataLocation::BOTH) {
        d_B_view = B.getGPUView();
        need_free_B = false;
    } else {
        d_B_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(B));
    }
#else
    d_A_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(A));
    d_B_view = ToDeviceMxVectorView(const_cast<MxVectorT&>(B));
#endif

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
    CUDA_CHECK_KERNEL();

    MxVectorViewT d_result_view = AllocateDeviceMxVectorView<BlockT>(A.Size());
    LaunchBatchedPackKernel<BlockViewT>(
        d_A_flat,
        d_result_view.blocks,
        d_result_view.numBlocks,
        elems_per_block);

    CUDA_CHECK(cudaFree(d_A_flat));
    CUDA_CHECK(cudaFree(d_B_flat));

    stream_pool.Release(compute_stream);

    if (need_free_A) {
        FreeDeviceMxVectorView(&d_A_view);
    }
    if (need_free_B) {
        FreeDeviceMxVectorView(&d_B_view);
    }

    MxVectorT result = ToHostMxVector<BlockT, MxVectorT>(d_result_view);
    FreeDeviceMxVectorView(&d_result_view);

    return result;
}

#ifndef JACOBI_KERNEL_INSTANTIATIONS_ONLY
// GPU implementation using normal CUDA arrays (f64/double)
void Jacobi2DGPUArrayNaive(const ElemType* A_host,
                          const ElemType* B_host,
                          ElemType* result_host,
                          uint32_t N,
                          uint32_t steps) {
    const size_t size = static_cast<size_t>(N) * N;
    const size_t bytes = size * sizeof(ElemType);
    
    // Allocate device memory
    ElemType* d_A = nullptr;
    ElemType* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_A, A_host, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B_host, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel configuration
    const int blockSize = 256;
    const int numBlocks = static_cast<int>((size + blockSize - 1) / blockSize);
    
    // Run Jacobi iterations
    for (uint32_t t = 0; t < steps; ++t) {
        Jacobi2DUpdateKernel<<<numBlocks, blockSize>>>(d_A, d_B, N);
        Jacobi2DUpdateKernel<<<numBlocks, blockSize>>>(d_B, d_A, N);
    }
    CUDA_CHECK_KERNEL();
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(result_host, d_A, bytes, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
}
#endif // JACOBI_KERNEL_INSTANTIATIONS_ONLY

