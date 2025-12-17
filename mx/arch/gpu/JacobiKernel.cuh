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
template <typename MxVectorT>
MxVectorT Jacobi2DGPUMxVectorNaive(const MxVectorT& A,
                                   const MxVectorT& B,
                                   uint32_t N,
                                   uint32_t steps);


