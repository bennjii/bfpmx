#pragma once
#include "../common.cuh"
#include "../cuda_utils.h"
#include <vector>
#include <cuda_runtime.h>

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
    int sizeBytes = mxvector.SizeInBytes();

    static constexpr uint8_t ExponentBits =  BlockT::FloatType::ExponentBits();
    static constexpr uint8_t SignificandBits = BlockT::FloatType::SignificandBits();
    static constexpr uint8_t SignBits = BlockT::FloatType::SignBits();
    using BlockViewT = BlockView<ExponentBits, SignificandBits, SignBits>;

    // Allocate GPU buffers
    BlockView<ExponentBits, SignificandBits, SignBits>* d_blocks_ptr = nullptr;
    // CUDA_CHECK(cudaMalloc(&d_blocks_ptr, mxvector.NumBlocks() * sizeof(BlockViewT)));
    d_blocks_ptr = (BlockView<ExponentBits, SignificandBits, SignBits>*)malloc(mxvector.NumBlocks() * sizeof(BlockViewT));
    for (int i = 0; i <  mxvector.NumBlocks(); i++) {
        BlockViewT blockView = ToDeviceBlockView(mxvector.BlockAt(i));
        // ✅ Properly copy struct to device memory
        CUDA_CHECK(cudaMemcpy(&d_blocks_ptr[i], 
                            &blockView, 
                            sizeof(BlockViewT), 
                            cudaMemcpyHostToHost));
    }

    // Fill MxVectorView
    MxVectorView<BlockT> view;
    view.blocks = d_blocks_ptr;
    view.numBlocks = mxvector.NumBlocks();
    view.numElems = mxvector.Size();
    return view;
}

template <typename BlockType, typename MxVectorT>
auto ToHostMxVector(const MxVectorView<BlockType>& mxvectorview) {
    using MxVectorViewT = MxVectorView<BlockType>;
    int sizeBytes = mxvectorview.numBlocks * sizeof(BlockType);

    static constexpr uint8_t ExponentBits = MxVectorViewT::ExponentBits;
    static constexpr uint8_t SignificandBits = MxVectorViewT::SignificandBits;
    static constexpr uint8_t SignBits = MxVectorViewT::SignBits;

    std::vector<BlockType> blocks(mxvectorview.numBlocks);
    // Memcpy
    CUDA_CHECK(cudaMemcpy((void*)blocks.data(),
               (void*)mxvectorview.blocks,
               sizeBytes,
               cudaMemcpyDeviceToHost));
    size_t num_blocks;
    return MxVectorT(blocks, mxvectorview.numElems);
}

template <typename MxVectorViewT>
void FreeDeviceMxVectorView(MxVectorViewT *v) {
    for (int i = 0; i < v->numBlocks; i++) {
        FreeDeviceBlockView(v->blocks[i]);
    }
    // CUDA_CHECK(cudaFree(v));
}

template <typename MxVectorT>
MxVectorT AddPointwiseGPUMxVectorNaive(const MxVectorT& a, const MxVectorT& b) {
    int blockSize = a.NumBlockElements() * sizeof(ElemType);
    size_t N = a.Size() * a.NumBlockElements();
    std::vector<ElemType> results (N);
    ElemType *d_l, *d_r, *d_result;
    cudaMalloc(&d_l, N * sizeof(ElemType));
    cudaMalloc(&d_r, N * sizeof(ElemType));
    cudaMalloc(&d_result, N * sizeof(ElemType));

    for (int i = 0; i < a.NumBlocks(); i++) {
        auto lhs = a.BlockAt(i);
        auto rhs = b.BlockAt(i);
        auto l = lhs.Spread();
        auto r = rhs.Spread();

        cudaMemcpy(d_l + i * a.NumBlockElements(), l.data(), blockSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r + i * b.NumBlockElements(), r.data(), blockSize, cudaMemcpyHostToDevice);
    }
    LaunchArithmeticKernel(d_l, d_r, d_result, N, ArithmeticOp::Add);
    for (int i = 0; i < a.NumBlocks(); i++) {
        cudaMemcpy(&results[i * a.NumBlockElements()], d_result + i * a.NumBlockElements(), blockSize, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_l);
    cudaFree(d_r);
    cudaFree(d_result);

    return MxVectorT(results);
}

template <typename MxVectorT>
MxVectorT AddPointwiseGPUMxVector(const MxVectorT& lhs, const MxVectorT& rhs) {
    using BlockT = MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    int sizeBytes = lhs.SizeInBytes();

    static constexpr uint8_t ExponentBits =  BlockT::FloatType::ExponentBits();
    static constexpr uint8_t SignificandBits = BlockT::FloatType::SignificandBits();
    static constexpr uint8_t SignBits = BlockT::FloatType::SignBits();
    using BlockViewT = BlockView<ExponentBits, SignificandBits, SignBits>;

    MxVectorViewT d_l, d_r, d_result;
    d_l = ToDeviceMxVectorView(lhs);
    d_r = ToDeviceMxVectorView(rhs);
    MxVectorT tmp = MxVectorT(lhs.Size());
    d_result = ToDeviceMxVectorView(tmp);
    // TODO: Implement arithmetic kernels here
    for (uint32_t i = 0; i < d_l.numBlocks; ++i) {
        using ElemType = f64;
        // std::cout <<  d_l.blocks[i].scalar<< '\n';
        const BlockViewT& l_block = d_l.blocks[i];
        const BlockViewT& r_block = d_r.blocks[i];
        BlockViewT& out_block = d_result.blocks[i]; // target block in final result

        const size_t n = lhs.NumBlockElements();

        ElemType* d_l_flat = nullptr;
        ElemType* d_r_flat = nullptr;
        ElemType* d_out_flat = nullptr;

        cudaMalloc(&d_l_flat,  n * sizeof(ElemType));
        cudaMalloc(&d_r_flat,  n * sizeof(ElemType));
        cudaMalloc(&d_out_flat, n * sizeof(ElemType));
    //     std::cout << n * sizeof(ElemType) << '\n';
    // std::cout << d_l_flat << ' ' << d_r_flat << ' ' << d_out_flat << "\n";
        LaunchSpreadKernel(&l_block, &r_block, d_l_flat, d_r_flat);
        LaunchArithmeticKernel(d_l_flat, d_r_flat, d_out_flat, n, ArithmeticOp::Add);
        LaunchPackKernel<typename MxVectorViewT::BlockViewT>(d_out_flat, &out_block);

        cudaFree(d_l_flat);
        cudaFree(d_r_flat);
        cudaFree(d_out_flat);
    }
    MxVectorT result = ToHostMxVector<BlockT, MxVectorT>(d_result);

    FreeDeviceMxVectorView(&d_l);
    FreeDeviceMxVectorView(&d_r);
    FreeDeviceMxVectorView(&d_result);

    return result;
}