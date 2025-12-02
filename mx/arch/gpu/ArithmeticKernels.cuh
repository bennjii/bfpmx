#pragma once
#include "common.cuh"

__global__ void AddKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n);
__global__ void SubKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n);
__global__ void MulKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n);
__global__ void DivKernel(const ElemType* l, const ElemType* r, ElemType* result, size_t n);

void LaunchArithmeticKernel(const ElemType* d_l,
                            const ElemType* d_r,
                            ElemType* d_out,
                            size_t n,
                            ArithmeticOp op);