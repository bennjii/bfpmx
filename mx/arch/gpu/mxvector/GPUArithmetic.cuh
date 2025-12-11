#pragma once
#include "../common.cuh"
#include "../cuda_utils.h"
#include "../SpreadKernel.cuh"
#include "../PackKernel.cuh"
#include "../ArithmeticKernels.cuh"
#include "../FusedKernel.cuh"
#include "../StreamPool.h"
#include "../PinnedMemoryPool.h"
#include "definition/vector/DataLocation.h"
#include <vector>
#include <cuda_runtime.h>
#include <cstring>
#include <memory>
#include "omp.h"

template <typename BlockType>
struct MxVectorView {
    static constexpr uint8_t ExponentBits = BlockType::FloatType::ExponentBits();
    static constexpr uint8_t SignificandBits = BlockType::FloatType::SignificandBits();
    static constexpr uint8_t SignBits = BlockType::FloatType::SignBits();
    using BlockViewT = BlockView<ExponentBits, SignificandBits, SignBits>;

    BlockViewT* blocks;
    uint32_t numBlocks;
    // Original number of elements, useful for when the number of elements is not divisible by block size
    uint32_t numElems;
};


template <typename MxVectorT>
auto ToDeviceMxVectorView(MxVectorT& mxvector) {
    using BlockT = MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;

    static constexpr uint8_t ExponentBits = BlockT::FloatType::ExponentBits();
    static constexpr uint8_t SignificandBits = BlockT::FloatType::SignificandBits();
    static constexpr uint8_t SignBits = BlockT::FloatType::SignBits();
    using BlockViewT = BlockView<ExponentBits, SignificandBits, SignBits>;

    constexpr size_t ElemSize = BlockT::FloatType::SizeBytes();
    constexpr size_t NumElems = BlockT::NumElems;
    const size_t num_blocks = mxvector.NumBlocks();
    const size_t block_data_size = NumElems * ElemSize;

    uint8_t* d_data_buffer = nullptr;
    BlockViewT* d_blocks_ptr = nullptr;
    std::vector<uint8_t> scalars(num_blocks);

    size_t total_data_size = num_blocks * block_data_size;
    size_t alignment = 256;
    size_t aligned_data_size = (total_data_size + alignment - 1) & ~(alignment - 1);
    CUDA_CHECK(cudaMalloc(&d_data_buffer, aligned_data_size));
    CUDA_CHECK(cudaMalloc(&d_blocks_ptr, num_blocks * sizeof(BlockViewT)));

    std::vector<BlockViewT> host_blocks(num_blocks);
    
    // Use pinned memory pool
    auto& mem_pool = GetGlobalPinnedMemoryPool();
    auto pinned_buffer = mem_pool.Acquire(total_data_size);
    uint8_t* pinned_data_buffer = static_cast<uint8_t*>(pinned_buffer->ptr);

    size_t offset = 0;
    for (size_t i = 0; i < num_blocks; ++i) {
        const BlockT& block = mxvector.BlockAt(i);
        const auto& block_data = block.data();
        const size_t block_bytes = NumElems * ElemSize;

        std::memcpy(pinned_data_buffer + offset, block_data.data(), block_bytes);
        scalars[i] = static_cast<uint8_t>(block.ScalarBits());

        host_blocks[i].data = d_data_buffer + offset;
        host_blocks[i].scalar = scalars[i];
        host_blocks[i].num_elems = NumElems;
        host_blocks[i].elem_size_bytes = ElemSize;

        offset += block_bytes;
    }

    // Use stream pool
    auto& stream_pool = GetGlobalStreamPool();
    cudaStream_t stream = stream_pool.Acquire();
    CUDA_CHECK(cudaMemcpyAsync(d_data_buffer, pinned_data_buffer,
                               total_data_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_blocks_ptr, host_blocks.data(),
                               num_blocks * sizeof(BlockViewT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stream_pool.Release(stream);
    
    // Release pinned memory back to pool
    mem_pool.Release(std::move(pinned_buffer));

    MxVectorView<BlockT> view;
    view.blocks = d_blocks_ptr;
    view.numBlocks = num_blocks;
    view.numElems = mxvector.Size();
    return view;
}

template <typename BlockType, typename MxVectorT>
auto ToHostMxVector(const MxVectorView<BlockType>& mxvectorview) {
    using MxVectorViewT = MxVectorView<BlockType>;
    using BlockViewT = typename MxVectorViewT::BlockViewT;

    constexpr size_t NumElems = BlockType::NumElems;
    constexpr size_t ElemSize = BlockType::FloatType::SizeBytes();
    const size_t block_data_size = NumElems * ElemSize;
    const size_t num_blocks = mxvectorview.numBlocks;
    const size_t total_data_size = num_blocks * block_data_size;

    std::vector<BlockViewT> host_block_views(num_blocks);
    
    // Use pinned memory pool
    auto& mem_pool = GetGlobalPinnedMemoryPool();
    auto pinned_buffer = mem_pool.Acquire(total_data_size);
    uint8_t* pinned_data_buffer = static_cast<uint8_t*>(pinned_buffer->ptr);
    
    // Use stream pool
    auto& stream_pool = GetGlobalStreamPool();
    cudaStream_t stream = stream_pool.Acquire();
    CUDA_CHECK(cudaMemcpyAsync(host_block_views.data(),
               mxvectorview.blocks,
               num_blocks * sizeof(BlockViewT),
               cudaMemcpyDeviceToHost, stream));
    
    if (num_blocks > 0) {
        CUDA_CHECK(cudaMemcpyAsync(pinned_data_buffer,
                   host_block_views[0].data,
                   total_data_size,
                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stream_pool.Release(stream);
    
    uint8_t* host_data_ptr = pinned_data_buffer;

    std::vector<BlockType> blocks(num_blocks);
    using PackedFloat = typename BlockType::PackedFloat;
    using ScalarType = typename BlockType::ScalarType;

    #pragma omp parallel for
    for (uint32_t i = 0; i < num_blocks; ++i) {
        const BlockViewT& view = host_block_views[i];
        const size_t offset = i * block_data_size;
        
        std::array<PackedFloat, NumElems> block_data;
        std::memcpy(block_data.data(),
                   host_data_ptr + offset,
                   block_data_size);

        ScalarType scalar = static_cast<ScalarType>(view.scalar);
        blocks[i] = BlockType(block_data, scalar);
    }

    // Release pinned memory back to pool
    mem_pool.Release(std::move(pinned_buffer));
    return MxVectorT(blocks, mxvectorview.numElems);
}

template <typename MxVectorViewT>
void FreeDeviceMxVectorView(MxVectorViewT *v) {
    if (v->blocks) {
        std::vector<typename MxVectorViewT::BlockViewT> host_blocks(v->numBlocks);
        CUDA_CHECK(cudaMemcpy(host_blocks.data(), v->blocks,
                             v->numBlocks * sizeof(typename MxVectorViewT::BlockViewT),
                             cudaMemcpyDeviceToHost));
        
        if (v->numBlocks > 0) {
            uint8_t* first_data_ptr = host_blocks[0].data;
            CUDA_CHECK(cudaFree(first_data_ptr));
        }
        CUDA_CHECK(cudaFree(v->blocks));
    }
}

template <typename BlockType>
MxVectorView<BlockType> AllocateDeviceMxVectorView(size_t num_elements) {
    using MxVectorViewT = MxVectorView<BlockType>;
    using BlockViewT = typename MxVectorViewT::BlockViewT;

    constexpr size_t ElemSize = BlockType::FloatType::SizeBytes();
    constexpr size_t NumElems = BlockType::NumElems;
    const size_t num_blocks = (num_elements + NumElems - 1) / NumElems;
    const size_t block_data_size = NumElems * ElemSize;
    const size_t total_data_size = num_blocks * block_data_size;

    size_t alignment = 256;
    size_t aligned_data_size = (total_data_size + alignment - 1) & ~(alignment - 1);
    
    uint8_t* d_data_buffer = nullptr;
    BlockViewT* d_blocks_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data_buffer, aligned_data_size));
    CUDA_CHECK(cudaMalloc(&d_blocks_ptr, num_blocks * sizeof(BlockViewT)));

    std::vector<BlockViewT> host_blocks(num_blocks);
    size_t offset = 0;
    for (size_t i = 0; i < num_blocks; ++i) {
        host_blocks[i].data = d_data_buffer + offset;
        host_blocks[i].scalar = 0;
        host_blocks[i].num_elems = NumElems;
        host_blocks[i].elem_size_bytes = ElemSize;
        offset += block_data_size;
    }

    auto& stream_pool = GetGlobalStreamPool();
    cudaStream_t stream = stream_pool.Acquire();
    CUDA_CHECK(cudaMemcpyAsync(d_blocks_ptr, host_blocks.data(),
                               num_blocks * sizeof(BlockViewT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stream_pool.Release(stream);

    MxVectorViewT view;
    view.blocks = d_blocks_ptr;
    view.numBlocks = num_blocks;
    view.numElems = num_elements;
    return view;
}

template <typename MxVectorT>
MxVectorT AddPointwiseGPUMxVectorNaive(const MxVectorT& a, const MxVectorT& b) {
    const size_t elems_per_block = a.NumBlockElements();
    const size_t num_blocks = a.NumBlocks();
    const size_t N = a.Size() * elems_per_block;
    const size_t block_size_bytes = elems_per_block * sizeof(ElemType);
    
    std::vector<ElemType> l_flat(N), r_flat(N), results(N);
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_blocks; ++i) {
        auto l = a.BlockAt(i).Spread();
        auto r = b.BlockAt(i).Spread();
        std::memcpy(l_flat.data() + i * elems_per_block, l.data(), block_size_bytes);
        std::memcpy(r_flat.data() + i * elems_per_block, r.data(), block_size_bytes);
    }

    ElemType *d_l, *d_r, *d_result;
    CUDA_CHECK(cudaMalloc(&d_l, N * sizeof(ElemType)));
    CUDA_CHECK(cudaMalloc(&d_r, N * sizeof(ElemType)));
    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(ElemType)));

    CUDA_CHECK(cudaMemcpy(d_l, l_flat.data(), N * sizeof(ElemType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, r_flat.data(), N * sizeof(ElemType), cudaMemcpyHostToDevice));
    
    LaunchArithmeticKernel(d_l, d_r, d_result, N, ArithmeticOp::Add);
    
    CUDA_CHECK(cudaMemcpy(results.data(), d_result, N * sizeof(ElemType), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_l));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_result));

    return MxVectorT(results);
}

template <typename MxVectorT>
MxVectorT AddPointwiseGPUMxVector(const MxVectorT& lhs, const MxVectorT& rhs) {
    using BlockT = MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;

    static constexpr uint8_t ExponentBits = BlockT::FloatType::ExponentBits();
    static constexpr uint8_t SignificandBits = BlockT::FloatType::SignificandBits();
    static constexpr uint8_t SignBits = BlockT::FloatType::SignBits();
    using BlockViewT = BlockView<ExponentBits, SignificandBits, SignBits>;

    MxVectorViewT d_l, d_r, d_result;
    d_l = ToDeviceMxVectorView(lhs);
    d_r = ToDeviceMxVectorView(rhs);
    MxVectorT tmp = MxVectorT(lhs.Size());
    d_result = ToDeviceMxVectorView(tmp);

    const uint32_t num_blocks = d_l.numBlocks;
    const uint32_t elems_per_block = lhs.NumBlockElements();
    const size_t total_elems = num_blocks * elems_per_block;

    ElemType* d_l_flat = nullptr;
    ElemType* d_r_flat = nullptr;
    ElemType* d_out_flat = nullptr;

    CUDA_CHECK(cudaMalloc(&d_l_flat, total_elems * sizeof(ElemType)));
    CUDA_CHECK(cudaMalloc(&d_r_flat, total_elems * sizeof(ElemType)));
    CUDA_CHECK(cudaMalloc(&d_out_flat, total_elems * sizeof(ElemType)));

    LaunchBatchedSpreadKernel<BlockViewT>(d_l.blocks, d_r.blocks, d_l_flat, d_r_flat, num_blocks, elems_per_block);
    LaunchArithmeticKernel(d_l_flat, d_r_flat, d_out_flat, total_elems, ArithmeticOp::Add);
    LaunchBatchedPackKernel<BlockViewT>(d_out_flat, d_result.blocks, num_blocks, elems_per_block);

    cudaFree(d_l_flat);
    cudaFree(d_r_flat);
    cudaFree(d_out_flat);

    MxVectorT result = ToHostMxVector<BlockT, MxVectorT>(d_result);

    FreeDeviceMxVectorView(&d_l);
    FreeDeviceMxVectorView(&d_r);
    FreeDeviceMxVectorView(&d_result);

    return result;
}

template <typename MxVectorT>
MxVectorT AddPointwiseGPUMxVectorFused(const MxVectorT& lhs, const MxVectorT& rhs) {
    using BlockT = MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;

    static constexpr uint8_t ExponentBits = BlockT::FloatType::ExponentBits();
    static constexpr uint8_t SignificandBits = BlockT::FloatType::SignificandBits();
    static constexpr uint8_t SignBits = BlockT::FloatType::SignBits();
    using BlockViewT = BlockView<ExponentBits, SignificandBits, SignBits>;

    auto& stream_pool = GetGlobalStreamPool();
    cudaStream_t compute_stream = stream_pool.Acquire();

    MxVectorViewT d_l, d_r;
    bool need_free_l = true;
    bool need_free_r = true;

    #ifdef HAS_CUDA
    if (lhs.getDataLocation() == mx::vector::DataLocation::GPU_ONLY || lhs.getDataLocation() == mx::vector::DataLocation::BOTH) {
        d_l = lhs.getGPUView();
        need_free_l = false;
    } else {
        d_l = ToDeviceMxVectorView(const_cast<MxVectorT&>(lhs));
    }

    if (rhs.getDataLocation() == mx::vector::DataLocation::GPU_ONLY || rhs.getDataLocation() == mx::vector::DataLocation::BOTH) {
        d_r = rhs.getGPUView();
        need_free_r = false;
    } else {
        d_r = ToDeviceMxVectorView(const_cast<MxVectorT&>(rhs));
    }
    #else
    d_l = ToDeviceMxVectorView(const_cast<MxVectorT&>(lhs));
    d_r = ToDeviceMxVectorView(const_cast<MxVectorT&>(rhs));
    #endif

    MxVectorViewT d_result = AllocateDeviceMxVectorView<BlockT>(lhs.Size());

    const uint32_t num_blocks = d_l.numBlocks;
    const uint32_t elems_per_block = lhs.NumBlockElements();

    LaunchFusedBlockArithmeticKernel<BlockViewT, ArithmeticOp::Add>(d_l.blocks, d_r.blocks, d_result.blocks, num_blocks, elems_per_block);

    stream_pool.Release(compute_stream);

    if (need_free_l) {
        FreeDeviceMxVectorView(&d_l);
    }
    if (need_free_r) {
        FreeDeviceMxVectorView(&d_r);
    }

    MxVectorT result(lhs.Size());
    result.setGPUView(d_result);

    return result;
}