#include "HeatKernel.cuh"

__global__ void Heat3DUpdateKernel(const ElemType* __restrict__ A,
                                   ElemType* __restrict__ B,
                                   uint32_t N) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(N) * N * N;
    if (idx >= total) return;

    const uint32_t i = idx / (N * N);
    const uint32_t rem = idx % (N * N);
    const uint32_t j = rem / N;
    const uint32_t k = rem % N;

    // Keep boundary cells unchanged
    if (i == 0 || j == 0 || k == 0 ||
        i >= N - 1 || j >= N - 1 || k >= N - 1) {
        B[idx] = A[idx];
        return;
    }

    const uint32_t center = idx;
    const uint32_t ip1 = (i + 1) * N * N + j * N + k;
    const uint32_t im1 = (i - 1) * N * N + j * N + k;
    const uint32_t jp1 = i * N * N + (j + 1) * N + k;
    const uint32_t jm1 = i * N * N + (j - 1) * N + k;
    const uint32_t kp1 = i * N * N + j * N + (k + 1);
    const uint32_t km1 = i * N * N + j * N + (k - 1);

    const double center_val = A[center];

    B[idx] = static_cast<ElemType>(
        0.125 * (A[ip1] - 2.0 * center_val + A[im1]) +
        0.125 * (A[jp1] - 2.0 * center_val + A[jm1]) +
        0.125 * (A[kp1] - 2.0 * center_val + A[km1]) +
        center_val
    );
}

// Naive Heat 3D over a flattened N×N×N grid stored in an MxVector.
// Keeps boundary cells unchanged. Returns the final A buffer after `steps`.
template <typename MxVectorT>
MxVectorT Heat3DGPUMxVectorNaive(const MxVectorT& A,
                                 const MxVectorT& B,
                                 uint32_t N,
                                 uint32_t steps) {
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    using BlockViewT = typename MxVectorViewT::BlockViewT;

    const uint64_t expected_size = static_cast<uint64_t>(N) * N * N;
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
    const uint64_t total_grid_elems = expected_size;
    const int numBlocksLaunch = static_cast<int>((total_grid_elems + blockSize - 1) / blockSize);

    for (uint32_t t = 0; t < steps; ++t) {
        Heat3DUpdateKernel<<<numBlocksLaunch, blockSize, 0, compute_stream>>>(
            d_A_flat, d_B_flat, N);
        Heat3DUpdateKernel<<<numBlocksLaunch, blockSize, 0, compute_stream>>>(
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

// Explicit instantiation for GPUVector used in JacobiHeatGPU tests
using GPUVector32Heat = mx::vector::MxVector<BlockDims<32>, unsigned char, fp8::E4M3Type,
                                             GPUArithmeticNaive, MaximumFractionalQuantization>;

template GPUVector32Heat Heat3DGPUMxVectorNaive<GPUVector32Heat>(
    const GPUVector32Heat&, const GPUVector32Heat&, uint32_t, uint32_t);

