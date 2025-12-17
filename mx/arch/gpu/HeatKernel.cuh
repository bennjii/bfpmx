#pragma once

#include "common.cuh"
#include "SpreadKernel.cuh"
#include "PackKernel.cuh"
#include "mxvector/GPUArithmetic.cuh"
#include "StreamPool.h"

// Device kernel (implemented in HeatKernel.cu)
__global__ void Heat3DUpdateKernel(const ElemType* __restrict__ A,
                                   ElemType* __restrict__ B,
                                   uint32_t N);

// Naive Heat 3D over a flattened N×N×N grid stored in an MxVector.
// Keeps boundary cells unchanged. Returns the final A buffer after `steps`.
template <typename MxVectorT>
MxVectorT Heat3DGPUMxVectorNaive(const MxVectorT& A,
                                 const MxVectorT& B,
                                 uint32_t N,
                                 uint32_t steps);


