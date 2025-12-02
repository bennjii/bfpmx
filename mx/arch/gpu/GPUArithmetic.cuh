#ifndef BFPMX_GPU_ARITHMETIC_CUH
#define BFPMX_GPU_ARITHMETIC_CUH
#include "common.cuh"
#include "ArithmeticKernels.cuh"
#include "SpreadKernel.cuh"
#include <cuda_runtime.h>
#include <iostream>

template <typename BlockT>
auto ToDeviceBlockView(const BlockT& block)
{
    constexpr int N = BlockT::NumElems;
    constexpr int ElemSize = BlockT::FloatType::SizeBytes();
    constexpr int ScalarBytes = BlockT::ScalarBytes;

    // Allocate GPU buffers
    uint8_t* d_data_ptr = nullptr;
    uint8_t d_scalar;
    uint8_t blockScalar = block.ScalarValue();

    cudaMalloc(&d_data_ptr, N * ElemSize);

    // Copy CPU quantized block → GPU memory
    cudaMemcpy(d_data_ptr,
               block.data().data(),
               N * ElemSize,
               cudaMemcpyHostToDevice);

    cudaMemcpy(&d_scalar,
               &blockScalar,
               sizeof(blockScalar),
               cudaMemcpyHostToDevice);

    // Fill BlockView
    using BV = BlockView<
        BlockT::FloatType::ExponentBits(),
        BlockT::FloatType::SignificandBits(),
        BlockT::FloatType::SignBits()
    >;

    BV view;
    view.data             = d_data_ptr;
    view.scalar           = d_scalar;
    view.num_elems        = N;
    view.elem_size_bytes  = ElemSize;

    return view;
}

template <typename BlockT>
inline void FreeDeviceBlockView(const BlockT& v)
{
    cudaFree((void*)v.data);
}

template <typename T>
struct GPUArithmeticNaive {
    // Only arithmetic operations are done on GPU, not Spread
    static T PointwiseOpNaive(const T& lhs, const T& rhs, auto op) {
        constexpr size_t N = T::Length();
        auto l = lhs.Spread();
        auto r = rhs.Spread();
        std::array<ElemType, N> result;

        ElemType *d_l, *d_r, *d_result;
        cudaMalloc(&d_l, N * sizeof(ElemType));
        cudaMalloc(&d_r, N * sizeof(ElemType));
        cudaMalloc(&d_result, N * sizeof(ElemType));

        cudaMemcpy(d_l, l.data(), N * sizeof(ElemType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, r.data(), N * sizeof(ElemType), cudaMemcpyHostToDevice);

        LaunchArithmeticKernel(d_l, d_r, d_result, N, op);

        cudaMemcpy(result.data(), d_result, N * sizeof(ElemType), cudaMemcpyDeviceToHost);

        cudaFree(d_l);
        cudaFree(d_r);
        cudaFree(d_result);

        return T(result);
    }
    static T Add(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Add); }
    static T Sub(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Sub); }
    static T Mul(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Mul); }
    static T Div(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Div); }
};


template <typename T>
struct GPUArithmetic {
    // Spread is done on GPU
    static T PointwiseOp(const T& lhs, const T& rhs, auto op) {
        // Untested, adjustable
        const int THRESHOLD = 0;
        // For small sizes, GPU overhead is larger than computation time
        if constexpr (T::Length() <= THRESHOLD) {
            return PointwiseOpNaive(lhs, rhs, op);
        } else {
            constexpr size_t N = T::Length();
            size_t blockSizeBytes = T::SizeBytes();
            size_t flattenSizeBytes = N * sizeof(ElemType);

            ElemType *d_l, *d_r, *d_result;
            cudaMalloc(&d_l, N * sizeof(ElemType));
            cudaMalloc(&d_r, N * sizeof(ElemType));
            cudaMalloc(&d_result, N * sizeof(ElemType));

            std::array<ElemType, N> result;

            const BlockView lhs_view = ToDeviceBlockView(lhs);
            const BlockView rhs_view = ToDeviceBlockView(rhs);

            LaunchSpreadKernel(&lhs_view, &rhs_view, d_l, d_r);
            LaunchArithmeticKernel(d_l, d_r, d_result, N, op);
            cudaMemcpy(result.data(), d_result, flattenSizeBytes, cudaMemcpyDeviceToHost);

            FreeDeviceBlockView(lhs_view);
            FreeDeviceBlockView(rhs_view);
            cudaFree(d_l);
            cudaFree(d_r);
            cudaFree(d_result);
            return T(result);
        }
    }
    static T Add(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Add); }
    static T Sub(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Sub); }
    static T Mul(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Mul); }
    static T Div(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Div); }
};

#endif
