#ifndef BFPMX_GPU_ARITHMETIC_H
#define BFPMX_GPU_ARITHMETIC_H

#include "definition/alias.h"
#include <array>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using ElemType = f64;
enum class ArithmeticOp : uint8_t {
    Add,
    Sub,
    Mul,
    Div
};
// Need a separate kernel because CUDA kernel launch can't be in cuh
void LaunchKernel(const ElemType* l, const ElemType* r, ElemType* out, size_t n, ArithmeticOp op);

template <typename T>
struct GPUArithmetic {
    static T PointwiseOp(const T& lhs, const T& rhs, auto op) {
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

        LaunchKernel(d_l, d_r, d_result, N, op);

        cudaMemcpy(result.data(), d_result, N * sizeof(ElemType), cudaMemcpyDeviceToHost);

        cudaFree(d_l);
        cudaFree(d_r);
        cudaFree(d_result);

        return T(result);
    }

    static T Add(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Add); }
    static T Sub(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Sub); }
    static T Mul(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Mul); }
    static T Div(const T& lhs, const T& rhs) { return PointwiseOp(lhs, rhs, ArithmeticOp::Div); }
};

#endif
