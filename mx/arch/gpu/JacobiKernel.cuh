#pragma once

#include "common.cuh"
#include "SpreadKernel.cuh"
#include "PackKernel.cuh"
#include "mxvector/GPUArithmetic.cuh"
#include "StreamPool.h"

// Device kernel (implemented in JacobiKernel.cu)
__global__ void Jacobi2DUpdateKernel(const ElemType* __restrict__ A,
                                     ElemType* __restrict__ B,
                                     uint32_t N);

// Naive Jacobi 2D over a flattened N×N grid stored in an MxVector.
// Keeps boundary cells unchanged. Returns the final A buffer after `steps`.
// Template definition is in JacobiKernel.cu (needs to be compiled with nvcc due to kernel launches)
template <typename MxVectorT>
MxVectorT Jacobi2DGPUMxVectorNaive(const MxVectorT& A,
                                   const MxVectorT& B,
                                   uint32_t N,
                                   uint32_t steps);

// GPU implementation using normal CUDA arrays (f64/double)
// Allocates device memory, runs kernel, and returns result in host array
void Jacobi2DGPUArrayNaive(const ElemType* A_host,
                          const ElemType* B_host,
                          ElemType* result_host,
                          uint32_t N,
                          uint32_t steps);


